"""Run Qwen3-VL-4B-Thinking on HJ/pseudo_test.json with thinking disabled.

Thinking control uses the backbone approach: after apply_chat_template, append
"\\n</think>\\n\\n" to close the thinking block before generation begins.
This is unconditional and does not depend on processor API support for
enable_thinking kwargs.
"""

import argparse
import base64
import csv
import difflib
import json
import math
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoConfig, AutoProcessor
except ImportError:
    AutoConfig = None
    AutoProcessor = None


DEFAULT_REPO_ROOT = Path("/home/super/hj")
DEFAULT_HJ_ROOT = DEFAULT_REPO_ROOT / "HJ"

if str(DEFAULT_HJ_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_HJ_ROOT))

# consolidated into wsi_latentmas_0701: resolve the moved wsi_vqa_baselines
# package (now at <repo>/wsi_vqa_baselines) without depending on the HJ symlink.
# The consolidated baseline package lives beside ``baselines`` under the
# single_v7 model directory.
_LATENTMAS_ROOT = Path(__file__).resolve().parent.parent
if str(_LATENTMAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_LATENTMAS_ROOT))

from wsi_vqa_baselines.prompting import (
    SYSTEM_PROMPT,
    count_text_tokens,
    format_vqa_prompt,
    parse_labeled_output as parse_baseline_labeled_output,
    select_choice_prediction as select_baseline_choice_prediction,
    strip_thinking as strip_baseline_thinking,
)
from wsi_vqa_baselines.prompting_bcnb import (
    SYSTEM_PROMPT_BCNB,
    format_bcnb_prompt,
)
from wsi_vqa_baselines.prompting_navigate import (
    format_navigate_prompt,
    parse_navigate_output,
)


def normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def mean(values):
    values = list(values)
    if not values:
        return float("nan")
    return sum(values) / len(values)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_pseudo_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def build_file_index(root, suffixes):
    index = {}
    if not root:
        return index
    root = Path(root)
    if not root.exists():
        return index
    for suffix in suffixes:
        for path in sorted(root.glob(suffix)):
            index.setdefault(path.stem, path)
            index.setdefault(path.name[:12], path)
    return index


def render_svs_thumbnail(slide_path, output_path, max_size):
    try:
        import openslide
    except ImportError as exc:
        raise RuntimeError(
            "openslide-python is required to render .svs thumbnails. "
            "Pass --image-root with pre-rendered PNG/JPG thumbnails if OpenSlide is unavailable."
        ) from exc

    slide = openslide.OpenSlide(str(slide_path))
    width, height = slide.dimensions
    scale = min(max_size / width, max_size / height, 1.0)
    thumb_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    image = slide.get_thumbnail(thumb_size).convert("RGB")
    image.save(output_path)


def resize_image_to_thumbnail(image_path, output_path, max_size):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_size, max_size))
    image.save(output_path)


def image_to_data_url(image_path):
    suffix = Path(image_path).suffix.lower()
    mime = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/png")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def prepare_images(samples, args):
    output_dir = Path(args.thumbnail_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    needed_slide_ids = sorted({str(sample["Id"]) for sample in samples})
    image_suffixes = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    slide_suffixes = ("*.svs", "*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")
    image_index = build_file_index(args.image_root, image_suffixes)
    slide_index = build_file_index(args.slide_dir, slide_suffixes)
    output_index = build_file_index(output_dir, image_suffixes)

    single_image = Path(args.image_path) if args.image_path else None
    if single_image is not None and not single_image.exists():
        raise FileNotFoundError(f"IMAGE_PATH does not exist: {single_image}")

    image_map = {}
    missing = []
    for slide_id in needed_slide_ids:
        existing = output_index.get(slide_id)
        if existing is not None and not args.overwrite_thumbnails:
            image_map[slide_id] = existing
            continue

        source = image_index.get(slide_id)
        if source is None and single_image is not None:
            source = single_image
        if source is None:
            source = slide_index.get(slide_id)
        if source is None:
            missing.append(slide_id)
            continue

        target = output_dir / f"{slide_id}.png"
        image_map[slide_id] = target
        if target.exists() and not args.overwrite_thumbnails:
            continue

        if source.suffix.lower() == ".svs":
            render_svs_thumbnail(source, target, args.thumbnail_size)
        else:
            resize_image_to_thumbnail(source, target, args.thumbnail_size)

    if missing:
        raise FileNotFoundError("Could not find image/slide files for IDs: " + ", ".join(missing))
    return image_map


def format_prompt(sample, force_choice_answer=True, mcq_style="letter", prompt_style="wsi"):
    question = str(sample.get("Question", "")).strip()
    choices = sample.get("Choice") or []
    if not isinstance(choices, list):
        choices = [choices]
    choices = [str(choice) for choice in choices]
    if prompt_style == "bcnb":
        # System line is delivered as a separate system message by the runner.
        return format_bcnb_prompt(question, choices, include_system_prompt=False)
    if prompt_style == "navigate":
        # PathNavigate-style prompt with JSON answer output. The per-sample
        # system block is embedded here (include_system_prompt=True); main()
        # neutralizes the separate system message to avoid double injection.
        return format_navigate_prompt(
            question,
            choices,
            force_choice_answer=force_choice_answer,
            mcq_style=mcq_style,
            include_system_prompt=True,
            use_guardrail=True,
        )
    return format_vqa_prompt(
        question,
        choices,
        force_choice_answer=force_choice_answer,
        mcq_style=mcq_style,
        include_system_prompt=False,
    )


def parse_labeled_output(text):
    return parse_baseline_labeled_output(text)


def clean_generation(text):
    text = str(text).strip()
    for marker in ("assistant\n", "assistant:", "Assistant:"):
        if marker in text:
            text = text.split(marker)[-1].strip()
    return text.strip()


def strip_thinking(text):
    return strip_baseline_thinking(text)


def strip_choice_prefix(text):
    text = str(text).strip()
    return re.sub(r"^\s*(?:answer\s*[:\-]\s*)?(?:option\s*)?([A-Z]|\d+)[\.\):\-]\s*", "", text, flags=re.I).strip()


def select_choice_prediction(raw_prediction, choices, mcq_style="letter"):
    return select_baseline_choice_prediction(raw_prediction, [str(choice) for choice in choices], mcq_style=mcq_style)


def choice_seq_correct(choices, ground_truth, prediction):
    if not choices:
        return None
    gt_score = difflib.SequenceMatcher(None, prediction, ground_truth).quick_ratio()
    for choice in choices:
        choice_score = difflib.SequenceMatcher(None, prediction, str(choice)).quick_ratio()
        if choice_score > gt_score:
            return False
    return True


def add_transformers_class_candidate(candidates, errors, class_name):
    if any(name == class_name for name, _ in candidates):
        return
    try:
        module = __import__("transformers", fromlist=[class_name])
        candidates.append((class_name, getattr(module, class_name)))
    except (ImportError, AttributeError) as exc:
        errors.append(f"{class_name} unavailable: {exc}")


def model_from_pretrained(cls, checkpoint, load_kwargs):
    try:
        return cls.from_pretrained(checkpoint, **load_kwargs)
    except TypeError as exc:
        if "dtype" not in load_kwargs:
            raise
        fallback_kwargs = dict(load_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        try:
            return cls.from_pretrained(checkpoint, **fallback_kwargs)
        except Exception:
            raise exc


def load_model(checkpoint, device_map, dtype_name, trust_remote_code, model_class=""):
    if torch is None or AutoConfig is None:
        raise RuntimeError("Local checkpoint mode requires torch and transformers.")

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype_name.lower())
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")

    load_kwargs = {
        "dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=trust_remote_code)
    model_type = str(getattr(config, "model_type", "")).lower()
    errors = []
    candidates = []

    if model_class:
        add_transformers_class_candidate(candidates, errors, model_class)
    else:
        for class_name in getattr(config, "architectures", None) or []:
            add_transformers_class_candidate(candidates, errors, class_name)

        try:
            from transformers import AutoModelForImageTextToText
            if not any(name == "AutoModelForImageTextToText" for name, _ in candidates):
                candidates.append(("AutoModelForImageTextToText", AutoModelForImageTextToText))
        except ImportError as exc:
            errors.append(f"AutoModelForImageTextToText unavailable: {exc}")

        if model_type.startswith("qwen3_vl"):
            add_transformers_class_candidate(candidates, errors, "Qwen3VLForConditionalGeneration")
        elif model_type.startswith("qwen3"):
            for class_name in ("Qwen3VLForConditionalGeneration", "Qwen3ForConditionalGeneration"):
                add_transformers_class_candidate(candidates, errors, class_name)
        else:
            try:
                from transformers import AutoModelForVision2Seq
                if not any(name == "AutoModelForVision2Seq" for name, _ in candidates):
                    candidates.append(("AutoModelForVision2Seq", AutoModelForVision2Seq))
            except ImportError as exc:
                errors.append(f"AutoModelForVision2Seq unavailable: {exc}")
            for class_name in ("Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"):
                add_transformers_class_candidate(candidates, errors, class_name)

    for name, cls in candidates:
        try:
            model = model_from_pretrained(cls, checkpoint, load_kwargs)
            model.eval()
            return model, name
        except Exception as exc:
            errors.append(f"{name} failed: {exc}")

    hint = ""
    if model_type.startswith("qwen3"):
        hint = (
            f"\nDetected model_type={model_type}. Install a transformers version that supports "
            "this Qwen3-VL checkpoint, or pass --model-class with the exact class name."
        )
    raise RuntimeError("Could not load Qwen-VL model." + hint + "\n" + "\n".join(errors))


class QwenVLRunner:
    """Local checkpoint runner using backbone-style thinking control.

    Thinking is disabled by appending "</think>" to the prompt after
    apply_chat_template, matching the approach in wsi_latentmas/backbone/qwen3vl.py.
    This is unconditional and does not depend on processor API kwargs.
    """

    def __init__(self, args):
        if AutoProcessor is None:
            raise RuntimeError("Local checkpoint mode requires transformers.")
        self.args = args
        self.processor = AutoProcessor.from_pretrained(
            args.checkpoint,
            trust_remote_code=args.trust_remote_code,
        )
        self.model, self.model_loader = load_model(
            args.checkpoint,
            args.device_map,
            args.torch_dtype,
            args.trust_remote_code,
            args.model_class,
        )
        try:
            from qwen_vl_utils import process_vision_info
            self.process_vision_info = process_vision_info
        except ImportError:
            self.process_vision_info = None

    def _close_thinking(self, text):
        """Backbone approach: append </think> to immediately close the thinking block."""
        return text + "\n</think>\n\n"

    def _inputs_with_qwen_utils(self, image_path, prompt):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Backbone-style: close thinking block before generation
        text = self._close_thinking(text)

        image_inputs, video_inputs = self.process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _inputs_fallback(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Backbone-style: close thinking block before generation
        text = self._close_thinking(text)

        return self.processor(text=[text], images=[image], padding=True, return_tensors="pt")

    def generate(self, image_path, prompt):
        if self.process_vision_info is not None:
            inputs = self._inputs_with_qwen_utils(image_path, prompt)
        else:
            inputs = self._inputs_fallback(image_path, prompt)
        inputs = inputs.to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": self.args.max_new_tokens,
            "num_beams": self.args.num_beams,
            "do_sample": self.args.do_sample,
            "repetition_penalty": self.args.repetition_penalty,
        }
        if self.args.do_sample and self.args.temperature > 0:
            generation_kwargs["temperature"] = self.args.temperature
        if self.args.do_sample and self.args.top_p is not None:
            generation_kwargs["top_p"] = self.args.top_p

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated_ids = generated_ids[:, input_ids.shape[-1]:]
        num_output_tokens = int(generated_ids.shape[-1])
        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return clean_generation(text), num_output_tokens


class OpenAICompatibleVLRunner:
    def __init__(self, args):
        self.args = args
        self.base_url = args.llm_url.rstrip("/")
        self.model = args.llm_model
        if not self.model:
            raise ValueError("--llm-model is required when --llm-url is used")

    def generate(self, image_path, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image_path)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature if self.args.do_sample else 0,
            # Disable thinking via API
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        if self.args.do_sample and self.args.top_p is not None:
            payload["top_p"] = self.args.top_p

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.args.api_key:
            headers["Authorization"] = f"Bearer {self.args.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.args.request_timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API request failed with HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not connect to LLM API at {self.base_url}: {exc}") from exc

        try:
            message = result["choices"][0]["message"]
            content = message.get("content") or ""
            num_output_tokens = int((result.get("usage") or {}).get("completion_tokens") or 0)
            return clean_generation(content), num_output_tokens
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM API response: {result}") from exc


def build_eval_rows(samples, predictions):
    rows = []
    for idx, sample in enumerate(samples):
        pred_row = predictions[str(idx)]
        choices = sample.get("Choice") or []
        if not isinstance(choices, list):
            choices = [choices]
        prediction = str(pred_row["prediction"])
        ground_truth = str(sample.get("Answer", ""))
        choice_correct = choice_seq_correct(choices, ground_truth, prediction)
        rows.append(
            {
                "question_id": str(idx),
                "slide_id": str(sample.get("Id", "")),
                "sample_index": idx,
                "question": sample.get("Question", ""),
                "prompt": pred_row.get("prompt", ""),
                "choices": choices,
                "prediction": prediction,
                "raw_prediction": pred_row.get("raw_prediction", prediction),
                "answer": pred_row.get("answer", prediction),
                "explanation": pred_row.get("explanation", ""),
                "choice_selection_method": pred_row.get("choice_selection_method", ""),
                "ground_truth": ground_truth,
                "image": pred_row.get("image", ""),
                "inference_time_sec": pred_row.get("inference_time_sec", ""),
                "num_output_tokens": pred_row.get("num_output_tokens", ""),
                "num_answer_tokens": pred_row.get("num_answer_tokens", count_text_tokens(None, pred_row.get("answer", ""))),
                "num_explanation_tokens": pred_row.get(
                    "num_explanation_tokens",
                    count_text_tokens(None, pred_row.get("explanation", "")),
                ),
                "exact_match": int(prediction == ground_truth),
                "normalized_exact_match": int(normalize_text(prediction) == normalize_text(ground_truth)),
                "has_choices": int(bool(choices)),
                "choice_seq_correct": "" if choice_correct is None else int(choice_correct),
            }
        )
    return rows


def write_predictions_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_vis_json(rows, vis_dir):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["slide_id"]].append(
            {
                "Question": row["question"],
                "Choice": row["choices"],
                "res": row["prediction"],
                "raw_res": row["raw_prediction"],
                "choice_selection_method": row["choice_selection_method"],
                "gts": row["ground_truth"],
                "image": row["image"],
                "inference_time_sec": row["inference_time_sec"],
            }
        )
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    for slide_id, items in grouped.items():
        with open(vis_dir / f"{slide_id}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)


def build_summary(rows):
    choice_values = [
        int(row["choice_seq_correct"])
        for row in rows
        if row["choice_seq_correct"] != ""
    ]
    inference_times = [
        float(row["inference_time_sec"])
        for row in rows
        if row["inference_time_sec"] != ""
    ]
    output_tokens = [
        int(row["num_output_tokens"])
        for row in rows
        if row.get("num_output_tokens") not in ("", None)
    ]
    answer_tokens = [
        int(row["num_answer_tokens"])
        for row in rows
        if row.get("num_answer_tokens") not in ("", None)
    ]
    explanation_tokens = [
        int(row["num_explanation_tokens"])
        for row in rows
        if row.get("num_explanation_tokens") not in ("", None)
    ]
    return {
        "num_samples": len(rows),
        "num_slides": len({row["slide_id"] for row in rows}),
        "exact_match": mean(row["exact_match"] for row in rows),
        "normalized_exact_match": mean(row["normalized_exact_match"] for row in rows),
        "num_choice_samples": len(choice_values),
        "choice_seq_accuracy": mean(choice_values),
        "total_inference_time_sec": sum(inference_times) if inference_times else float("nan"),
        "avg_inference_time_sec_per_question": mean(inference_times),
        "total_output_tokens": sum(output_tokens),
        "avg_output_tokens_per_question": mean(output_tokens),
        "total_answer_tokens": sum(answer_tokens),
        "avg_answer_tokens_per_question": mean(answer_tokens),
        "total_explanation_tokens": sum(explanation_tokens),
        "avg_explanation_tokens_per_question": mean(explanation_tokens),
    }


def write_details_csv(rows, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "slide_id",
        "sample_index",
        "question",
        "prompt",
        "choices_json",
        "prediction",
        "answer",
        "explanation",
        "raw_prediction",
        "choice_selection_method",
        "ground_truth",
        "image",
        "inference_time_sec",
        "num_output_tokens",
        "num_answer_tokens",
        "num_explanation_tokens",
        "exact_match",
        "normalized_exact_match",
        "has_choices",
        "choice_seq_correct",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["choices_json"] = json.dumps(csv_row.pop("choices"), ensure_ascii=False)
            writer.writerow(csv_row)


def write_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: (None if isinstance(value, float) and math.isnan(value) else value)
        for key, value in payload.items()
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def write_general_eval_jsonl(rows, output_jsonl):
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            item = {
                "question_id": row["question_id"],
                "slide_id": row["slide_id"],
                "question": row["question"],
                "prompt": row["prompt"],
                "prediction": row["prediction"],
                "raw_prediction": row["raw_prediction"],
                "choice_selection_method": row["choice_selection_method"],
                "ground_truth": row["ground_truth"],
                "choices": row["choices"],
                "is_mcq": bool(row["choices"]),
                "image": row["image"],
                "inference_time_sec": row["inference_time_sec"],
                "num_output_tokens": row["num_output_tokens"],
                "num_answer_tokens": row["num_answer_tokens"],
                "num_explanation_tokens": row["num_explanation_tokens"],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-4B-Thinking (thinking disabled) on pseudo_test.json."
    )
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--llm-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--input-json", default=str(DEFAULT_HJ_ROOT / "pseudo_test.json"))
    parser.add_argument("--slide-dir", default=str(DEFAULT_HJ_ROOT / "tcga_svs_flat"))
    parser.add_argument("--image-root", default="")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--thumbnail-dir", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--vis-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--general-jsonl", required=True)
    parser.add_argument("--inference-time-json", required=True)
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--model-class", default="")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--overwrite-thumbnails", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--no-force-choice-answer", action="store_true")
    parser.add_argument("--mcq-style", choices=["text", "letter"], default="letter",
                        help="text: return exact choice text; letter: return only A/B/C/D")
    parser.add_argument("--prompt-style", choices=["wsi", "bcnb", "navigate"], default="wsi",
                        help="wsi: default labeled answer/explanation prompt; "
                             "bcnb: terse letter-answer MCQ prompt (SlideBench-VQA-BCNB style); "
                             "navigate: PathNavigate-style prompt with JSON answer output")
    parser.add_argument("--print-out", action="store_true")
    return parser.parse_args()


def make_progress(total, initial):
    """tqdm bar written to the real terminal (/dev/tty) so it stays visible even
    when the launcher redirects stdout/stderr to a log file. Returns None if tqdm
    is unavailable."""
    try:
        from tqdm import tqdm
    except ImportError:
        return None
    try:
        stream = open("/dev/tty", "w")
    except OSError:
        stream = sys.stderr
    return tqdm(total=total, initial=initial, file=stream, dynamic_ncols=True, desc="inference")


def predictions_complete(path, expected_count):
    path = Path(path)
    if not path.exists():
        return False
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count == expected_count


def load_existing_predictions(path):
    rows = {}
    path = Path(path)
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Tolerate a truncated final line left by an interrupted run.
                continue
            qid = row.get("question_id")
            if qid is not None:
                rows[str(qid)] = row
    return rows


def main():
    args = parse_args()
    set_seed(args.seed)
    if not args.llm_url and not args.checkpoint:
        raise SystemExit("--checkpoint is required unless --llm-url is provided")
    # BCNB style: terse letter-answer MCQ. Swap the system message used by the
    # runner, and score by mapping the predicted letter back to the choice text.
    if args.prompt_style == "bcnb":
        globals()["SYSTEM_PROMPT"] = SYSTEM_PROMPT_BCNB
    elif args.prompt_style == "navigate":
        # Navigate embeds its (per-sample) system block in the user prompt, so
        # send an empty separate system message.
        globals()["SYSTEM_PROMPT"] = ""
    select_mcq_style = "text" if args.prompt_style == "bcnb" else args.mcq_style
    samples = load_pseudo_json(args.input_json)
    image_map = prepare_images(samples, args)

    if args.skip_inference or (
        predictions_complete(args.answers_file, len(samples)) and not args.overwrite_predictions
    ):
        predictions = load_existing_predictions(args.answers_file)
        elapsed = sum(float(row.get("inference_time_sec") or 0.0) for row in predictions.values())
        total_output_tokens = sum(int(row.get("num_output_tokens") or 0) for row in predictions.values())
    else:
        if args.llm_url:
            runner = OpenAICompatibleVLRunner(args)
            print(f"Using OpenAI-compatible VL API: {args.llm_url} model={args.llm_model}")
        else:
            runner = QwenVLRunner(args)
            print(f"Loaded model with {runner.model_loader}")
        predictions = {}
        if not args.overwrite_predictions:
            predictions = load_existing_predictions(args.answers_file)
            if predictions:
                print(f"Resuming from {len(predictions)}/{len(samples)} existing predictions")
        Path(args.answers_file).parent.mkdir(parents=True, exist_ok=True)
        pred_fh = open(args.answers_file, "a" if predictions else "w", encoding="utf-8")
        progress = make_progress(len(samples), len(predictions))
        total_started = time.time()
        for idx, sample in enumerate(samples):
            question_id = str(idx)
            if question_id in predictions:
                continue
            prompt = format_prompt(sample, force_choice_answer=not args.no_force_choice_answer, mcq_style=args.mcq_style, prompt_style=args.prompt_style)
            choices = sample.get("Choice") or []
            if not isinstance(choices, list):
                choices = [choices]
            started = time.time()
            raw_prediction, num_output_tokens = runner.generate(image_map[str(sample["Id"])], prompt)
            # Strip any residual <think> blocks as a safety net
            raw_prediction = strip_thinking(raw_prediction)
            latency = time.time() - started
            if args.prompt_style == "bcnb":
                # Letter-answer prompt: no answer/explanation labels.
                answer, explanation = raw_prediction.strip(), ""
            elif args.prompt_style == "navigate":
                # JSON answer output (PathNavigate-style); falls back to labeled.
                answer, explanation = parse_navigate_output(raw_prediction)
            else:
                # Parse labeled format: "answer: ...\nexplanation: ..."
                answer, explanation = parse_labeled_output(raw_prediction)
            tokenizer = getattr(getattr(runner, "processor", None), "tokenizer", None)
            num_answer_tokens = count_text_tokens(tokenizer, answer)
            num_explanation_tokens = count_text_tokens(tokenizer, explanation)
            if choices and not args.no_force_choice_answer:
                prediction, choice_selection_method = select_choice_prediction(answer, choices, select_mcq_style)
            else:
                prediction = answer
                choice_selection_method = "open_ended" if not choices else "force_choice_disabled"
            predictions[question_id] = {
                "question_id": question_id,
                "slide_id": str(sample.get("Id", "")),
                "question": sample.get("Question", ""),
                "prompt": prompt,
                "choices": choices,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "answer": answer,
                "explanation": explanation,
                "choice_selection_method": choice_selection_method,
                "ground_truth": sample.get("Answer", ""),
                "image": str(image_map[str(sample["Id"])]),
                "inference_time_sec": latency,
                "num_output_tokens": num_output_tokens,
                "num_answer_tokens": num_answer_tokens,
                "num_explanation_tokens": num_explanation_tokens,
            }
            pred_fh.write(json.dumps(predictions[question_id], ensure_ascii=False) + "\n")
            pred_fh.flush()
            if progress is not None:
                progress.update(1)
            if args.print_out:
                print(f"[{idx + 1}/{len(samples)}] {sample.get('Id')} -> {prediction} | {explanation}")
        pred_fh.close()
        if progress is not None:
            progress.close()
        elapsed = sum(float(row.get("inference_time_sec") or 0.0) for row in predictions.values())
        total_output_tokens = sum(int(row.get("num_output_tokens") or 0) for row in predictions.values())

    rows = build_eval_rows(samples, predictions)
    summary = build_summary(rows)

    write_predictions_jsonl(rows, args.answers_file)
    write_vis_json(rows, args.vis_dir)
    write_details_csv(rows, args.output_csv)
    write_json(summary, args.output_json)
    write_general_eval_jsonl(rows, args.general_jsonl)
    write_json(
        {
            "inference_time_sec": elapsed,
            "num_predictions": len(samples),
            "total_output_tokens": total_output_tokens,
            "total_answer_tokens": sum(int(row.get("num_answer_tokens") or 0) for row in rows),
            "total_explanation_tokens": sum(int(row.get("num_explanation_tokens") or 0) for row in rows),
        },
        args.inference_time_json,
    )

    print(f"Answers: {args.answers_file}")
    print(f"Vis JSON: {args.vis_dir}")
    print(f"Details CSV: {args.output_csv}")
    print(f"Summary JSON: {args.output_json}")
    print(f"General eval JSONL: {args.general_jsonl}")
    print(f"Inference time JSON: {args.inference_time_json}")
    print(f"total_output_tokens: {total_output_tokens}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
