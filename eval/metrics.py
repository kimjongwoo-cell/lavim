# #!/usr/bin/env python
# """General WSI-VQA evaluator for model outputs under latentWSI.

# This script normalizes common WSI-VQA output formats into rows with:
# question_id, question, prediction, ground_truth, choices, inference_time_sec.

# It computes the same core NLU metrics used by:
# - HJ/PathAgent/eval/metrics_id_only.py
# - HJ/WSI-VQA/modules/metrics.py

# It also includes general VQA utility metrics that are useful across models:
# - PathAgent-style MCQ sequence accuracy
# - normalized exact match
# - token F1
# - optional survival-time C-index when applicable
# """

# import argparse
# import csv
# import difflib
# import glob
# import hashlib
# import json
# import math
# import os
# import re
# import string
# from collections import Counter
# from dataclasses import dataclass
# from typing import Any, Dict, Iterable, List, Optional, Tuple


# LETTER_RE = re.compile(r"\b([A-Z])\b")
# CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])[\.\):]\s*(.+?)\s*$")
# INLINE_CHOICE_RE = re.compile(r"(?:^|[;\n]\s*)([A-Z])[\.\):]\s*([^;\n]+)")
# CHOICE_MARKER_RE = re.compile(r"(?:^|\s)([A-H])\.\s*")


# @dataclass
# class EvalRow:
#     source_file: str
#     question_id: str
#     long_id: str
#     question: str
#     prediction: str
#     ground_truth: str
#     choices: List[str]
#     is_mcq: bool = False
#     metadata: str = ""
#     inference_time_sec: Optional[float] = None


# def parse_args():
#     parser = argparse.ArgumentParser(description="General evaluator for WSI-VQA model outputs.")
#     parser.add_argument(
#         "--input",
#         nargs="+",
#         default=[],
#         help="Output files or directories to evaluate. Directories are scanned recursively.",
#     )
#     parser.add_argument(
#         "--glob",
#         action="append",
#         default=[],
#         help="Glob pattern for output files. Can be passed more than once.",
#     )
#     parser.add_argument(
#         "--discover-root",
#         default="",
#         help="Scan this root for likely model output JSON/JSONL files only.",
#     )
#     parser.add_argument(
#         "--questions",
#         default="",
#         help="Optional question/GT JSONL used to fill missing GT by question_id.",
#     )
#     parser.add_argument(
#         "--output-dir",
#         default="eval/results",
#         help="Directory for metrics JSON/CSV outputs.",
#     )
#     parser.add_argument(
#         "--name",
#         default="general_wsi_vqa",
#         help="Run name used for output filenames.",
#     )
#     parser.add_argument(
#         "--inference-time-json",
#         default="",
#         help="Optional JSON file containing inference_time_sec and num_predictions.",
#     )
#     parser.add_argument(
#         "--inference-seconds",
#         type=float,
#         default=None,
#         help="Optional total inference wall time in seconds.",
#     )
#     parser.add_argument(
#         "--recursive",
#         action="store_true",
#         help="Recursively scan directories from --input. Enabled automatically for directories.",
#     )
#     parser.add_argument(
#         "--metric-scope",
#         choices=["split", "all", "both"],
#         default="both",
#         help=(
#             "split: report COCO text metrics on open-ended only, with MCQ accuracy separately. "
#             "all: report COCO text metrics on every valid sample, like HJ/WSI-VQA/modules/metrics.py. "
#             "both: save both sets; default summary BLEU/METEOR/ROUGE use split/open-ended."
#         ),
#     )
#     return parser.parse_args()


# def read_json(path: str) -> Any:
#     with open(path, "r", encoding="utf-8") as handle:
#         return json.load(handle)


# def read_jsonl(path: str) -> List[Dict[str, Any]]:
#     rows = []
#     with open(path, "r", encoding="utf-8") as handle:
#         for line in handle:
#             if line.strip():
#                 rows.append(json.loads(line))
#     return rows


# def normalize_text(text: Any) -> str:
#     text = str(text).lower().strip()
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     return " ".join(text.split())


# def token_f1(prediction: str, answer: str) -> float:
#     pred_tokens = normalize_text(prediction).split()
#     answer_tokens = normalize_text(answer).split()
#     if not pred_tokens and not answer_tokens:
#         return 1.0
#     if not pred_tokens or not answer_tokens:
#         return 0.0

#     common = Counter(pred_tokens) & Counter(answer_tokens)
#     num_same = sum(common.values())
#     if num_same == 0:
#         return 0.0
#     precision = num_same / len(pred_tokens)
#     recall = num_same / len(answer_tokens)
#     return 2 * precision * recall / (precision + recall)


# def metric_tokenize(text: str) -> List[str]:
#     return re.findall(r"\w+|[^\w\s]", str(text).lower())


# def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
#     return [tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)]


# def fallback_bleu(gts: Dict[str, List[str]], res: Dict[str, List[str]], max_n: int = 4) -> List[float]:
#     clipped = [0] * max_n
#     totals = [0] * max_n
#     ref_len = 0
#     pred_len = 0

#     for key, refs in gts.items():
#         ref_tokens = metric_tokenize(refs[0])
#         pred_tokens = metric_tokenize(res[key][0])
#         ref_len += len(ref_tokens)
#         pred_len += len(pred_tokens)
#         for n in range(1, max_n + 1):
#             pred_counts = Counter(ngrams(pred_tokens, n))
#             ref_counts = Counter(ngrams(ref_tokens, n))
#             clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
#             totals[n - 1] += sum(pred_counts.values())

#     if pred_len == 0:
#         return [0.0] * max_n
#     bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
#     scores = []
#     for n in range(1, max_n + 1):
#         precisions = [(clipped[idx] + 1.0) / (totals[idx] + 1.0) for idx in range(n)]
#         scores.append(bp * math.exp(sum(math.log(p) for p in precisions) / n))
#     return scores


# def lcs_len(a: List[str], b: List[str]) -> int:
#     prev = [0] * (len(b) + 1)
#     for token_a in a:
#         cur = [0] * (len(b) + 1)
#         for idx, token_b in enumerate(b, 1):
#             cur[idx] = prev[idx - 1] + 1 if token_a == token_b else max(prev[idx], cur[idx - 1])
#         prev = cur
#     return prev[-1]


# def fallback_rouge_l(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> float:
#     scores = []
#     for key, refs in gts.items():
#         ref_tokens = metric_tokenize(refs[0])
#         pred_tokens = metric_tokenize(res[key][0])
#         if not ref_tokens or not pred_tokens:
#             scores.append(0.0)
#             continue
#         lcs = lcs_len(ref_tokens, pred_tokens)
#         precision = lcs / len(pred_tokens)
#         recall = lcs / len(ref_tokens)
#         scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
#     return sum(scores) / len(scores) if scores else 0.0


# def fallback_coco_scores(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
#     bleu = fallback_bleu(gts, res)
#     return {
#         "BLEU_1": bleu[0],
#         "BLEU_2": bleu[1],
#         "BLEU_3": bleu[2],
#         "BLEU_4": bleu[3],
#         "METEOR": 0.0,
#         "ROUGE_L": fallback_rouge_l(gts, res),
#     }


# def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
#     clean = [value for value in values if value is not None and not math.isnan(value)]
#     if not clean:
#         return None
#     return sum(clean) / len(clean)


# def make_unique_id(long_id: str, question_text: str) -> str:
#     q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
#     return f"{long_id}_{q_hash}"


# def first_present(data: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
#     for key in keys:
#         value = data.get(key)
#         if value is not None and value != "":
#             return value
#     return default


# def as_text(value: Any) -> str:
#     if isinstance(value, list):
#         return str(value[0]) if value else ""
#     if value is None:
#         return ""
#     return str(value)


# def parse_question_choices(question: Any) -> List[str]:
#     choices = []
#     text = as_text(question)
#     for line in text.splitlines():
#         match = CHOICE_LINE_RE.match(line)
#         if match:
#             choices.append(match.group(2).strip())
#     if choices:
#         return choices

#     for _, choice in INLINE_CHOICE_RE.findall(text):
#         choice = choice.strip()
#         if choice.lower().startswith("answer with"):
#             continue
#         choices.append(choice)
#     return choices


# def parse_marker_choices(value: Any) -> List[str]:
#     text = as_text(value)
#     if not text.strip():
#         return []
#     matches = list(CHOICE_MARKER_RE.finditer(text))
#     if not matches:
#         return []
#     choices = []
#     for idx, match in enumerate(matches):
#         end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
#         choice = text[match.end() : end].strip()
#         choice = re.sub(r"\n?Answer with the best choice\.?\s*$", "", choice, flags=re.I).strip()
#         if choice:
#             choices.append(choice)
#     return choices


# def coerce_choices(data: Dict[str, Any]) -> List[str]:
#     raw = first_present(data, ["choices", "Choice", "choices_json"], default=[])
#     if isinstance(raw, str):
#         raw = raw.strip()
#         if not raw:
#             return parse_question_choices(first_present(data, ["question", "Question"], default=""))
#         try:
#             parsed = json.loads(raw)
#             raw = parsed
#         except json.JSONDecodeError:
#             parsed_choices = parse_marker_choices(raw)
#             if parsed_choices:
#                 return parsed_choices
#             raw = [raw]
#     if raw is None:
#         return parse_question_choices(first_present(data, ["question", "Question"], default=""))
#     if isinstance(raw, dict):
#         return [str(raw[key]).strip() for key in sorted(raw) if str(raw[key]).strip()]
#     if isinstance(raw, list):
#         choices = [as_text(item).strip() for item in raw if as_text(item).strip()]
#         return choices or parse_question_choices(first_present(data, ["question", "Question"], default=""))
#     return [str(raw).strip()] if str(raw).strip() else []


# def infer_is_mcq(data: Dict[str, Any], choices: List[str]) -> bool:
#     if choices:
#         return True
#     value = data.get("is_mcq")
#     if isinstance(value, bool):
#         return value
#     if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "closed", "mcq"}:
#         return True
#     answer_type = as_text(first_present(data, ["answer_type", "metadata", "type"], default="")).strip().lower()
#     return answer_type in {"closed", "close-ended", "closed-ended", "multiple-choice", "multiple_choice", "mcq"}


# def parse_inference_time(data: Dict[str, Any]) -> Optional[float]:
#     value = first_present(
#         data,
#         ["inference_time_sec", "inference_time", "elapsed_sec", "time_sec", "latency_sec"],
#         default=None,
#     )
#     if value is None or value == "":
#         return None
#     try:
#         return float(value)
#     except (TypeError, ValueError):
#         return None


# def row_from_dict(data: Dict[str, Any], source_file: str, fallback_id: str) -> Optional[EvalRow]:
#     question = as_text(first_present(data, ["raw_question", "question", "Question", "prompt"], default=""))
#     if "res" in data:
#         prediction_value = first_present(data, ["res"], default="")
#     elif "T-answer" in data or "ground_truth" in data or "gt" in data:
#         prediction_value = first_present(
#             data,
#             ["pred_answer", "prediction", "Output", "output", "answer", "text", "response", "res"],
#             default="",
#         )
#     else:
#         prediction_value = first_present(
#             data,
#             ["pred_answer", "prediction", "Output", "output", "answer", "text", "response", "res"],
#             default="",
#         )
#     prediction = as_text(prediction_value).strip()
#     if prediction == question and "text" in data:
#         prediction = ""
#     ground_truth = as_text(
#         first_present(data, ["ground_truth", "T-answer", "gts", "gt", "answer_gt", "target", "label", "Answer"], default="")
#     ).strip()
#     question_id = as_text(first_present(data, ["question_id", "id", "Id"], default="")).strip()
#     long_id = as_text(first_present(data, ["long_id", "wsi_full_id", "slide_id", "case_id", "image", "Id"], default="")).strip()
#     metadata = as_text(first_present(data, ["metadata", "answer_type"], default="")).strip()
#     choices = coerce_choices(data)
#     is_mcq = infer_is_mcq(data, choices)

#     if not question_id:
#         question_id = make_unique_id(long_id, question) if long_id and question else fallback_id

#     if not prediction and not ground_truth:
#         return None

#     return EvalRow(
#         source_file=source_file,
#         question_id=question_id,
#         long_id=long_id,
#         question=question,
#         prediction=prediction,
#         ground_truth=ground_truth,
#         choices=choices,
#         is_mcq=is_mcq,
#         metadata=metadata,
#         inference_time_sec=parse_inference_time(data),
#     )


# def rows_from_json_payload(payload: Any, source_file: str) -> List[EvalRow]:
#     if isinstance(payload, list):
#         rows = []
#         for idx, item in enumerate(payload):
#             if isinstance(item, dict):
#                 row = row_from_dict(item, source_file, f"{os.path.basename(source_file)}_{idx}")
#                 if row:
#                     rows.append(row)
#         return rows

#     if isinstance(payload, dict):
#         if all(isinstance(value, list) for value in payload.values()):
#             rows = []
#             for case_id, samples in payload.items():
#                 for idx, item in enumerate(samples):
#                     if isinstance(item, dict):
#                         merged = dict(item)
#                         merged.setdefault("case_id", case_id)
#                         row = row_from_dict(merged, source_file, f"{case_id}_{idx}")
#                         if row:
#                             rows.append(row)
#             return rows

#         row = row_from_dict(payload, source_file, os.path.splitext(os.path.basename(source_file))[0])
#         return [row] if row else []

#     return []


# def load_rows_from_file(path: str) -> List[EvalRow]:
#     if path.endswith(".jsonl"):
#         rows = []
#         for idx, item in enumerate(read_jsonl(path)):
#             row = row_from_dict(item, path, f"{os.path.basename(path)}_{idx}")
#             if row:
#                 rows.append(row)
#         return rows

#     if path.endswith(".json"):
#         return rows_from_json_payload(read_json(path), path)

#     return []


# def is_likely_output_file(path: str) -> bool:
#     lower = path.lower()
#     include_tokens = ["output", "outputs", "result", "results", "answer", "answers", "pred", "prediction", "vis"]
#     exclude_tokens = [
#         "/data/",
#         "/playground/data/",
#         "/checkpoints/",
#         "/checkpoint/",
#         "/models/",
#         "/model/",
#         "/cache/",
#         "__pycache__",
#         "/patches_output/",
#     ]
#     return any(token in lower for token in include_tokens) and not any(token in lower for token in exclude_tokens)


# def discover_output_files(root: str) -> List[str]:
#     if not root:
#         return []
#     files = []
#     for pattern in ["**/*.json", "**/*.jsonl"]:
#         for path in glob.glob(os.path.join(root, pattern), recursive=True):
#             if os.path.isfile(path) and is_likely_output_file(path):
#                 files.append(path)
#     return files


# def collect_files(inputs: List[str], patterns: List[str], discover_root: str = "") -> List[str]:
#     files = []
#     for item in inputs:
#         if os.path.isdir(item):
#             files.extend(
#                 path
#                 for path in glob.glob(os.path.join(item, "**", "*"), recursive=True)
#                 if path.endswith((".json", ".jsonl"))
#             )
#         elif os.path.isfile(item) and item.endswith((".json", ".jsonl")):
#             files.append(item)

#     for pattern in patterns:
#         files.extend(path for path in glob.glob(pattern, recursive=True) if path.endswith((".json", ".jsonl")))

#     files.extend(discover_output_files(discover_root))

#     unique = []
#     seen = set()
#     for path in files:
#         abs_path = os.path.abspath(path)
#         if abs_path not in seen:
#             seen.add(abs_path)
#             unique.append(path)
#     return sorted(unique)


# def load_question_gt(path: str) -> Dict[str, Dict[Any, EvalRow]]:
#     if not path:
#         return {}
#     rows = load_rows_from_file(path)
#     by_id = {row.question_id: row for row in rows if row.question_id}
#     by_pair = {
#         (row.long_id, row.question): row
#         for row in rows
#         if row.long_id and row.question
#     }
#     return {"by_id": by_id, "by_pair": by_pair}


# def fill_missing_gt(rows: List[EvalRow], references: Dict[str, Dict[Any, EvalRow]]) -> List[EvalRow]:
#     if not references:
#         return rows

#     filled = []
#     for row in rows:
#         ref = references.get("by_id", {}).get(row.question_id)
#         if ref is None and row.long_id and row.question:
#             ref = references.get("by_pair", {}).get((row.long_id, row.question))
#         if ref and not row.ground_truth:
#             row.ground_truth = ref.ground_truth
#         if ref and not row.question:
#             row.question = ref.question
#         if ref and not row.choices:
#             row.choices = ref.choices
#         if ref and not row.is_mcq:
#             row.is_mcq = ref.is_mcq
#         filled.append(row)
#     return filled


# def compute_coco_scores(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Tuple[Dict[str, float], str]:
#     empty_scores = {
#         "BLEU_1": 0.0,
#         "BLEU_2": 0.0,
#         "BLEU_3": 0.0,
#         "BLEU_4": 0.0,
#         "METEOR": 0.0,
#         "ROUGE_L": 0.0,
#     }
#     if not gts:
#         return empty_scores, ""
#     try:
#         from pycocoevalcap.bleu.bleu import Bleu
#         from pycocoevalcap.meteor.meteor import Meteor
#         from pycocoevalcap.rouge.rouge import Rouge
#     except ModuleNotFoundError as exc:
#         return fallback_coco_scores(gts, res), f"pycocoevalcap unavailable; using fallback BLEU/ROUGE and METEOR=0: {exc}"

#     scorers = [
#         (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
#         (Meteor(), "METEOR"),
#         (Rouge(), "ROUGE_L"),
#     ]
#     scores = {}
#     for scorer, method in scorers:
#         score, _ = scorer.compute_score(gts, res)
#         if isinstance(method, list):
#             for sc, name in zip(score, method):
#                 scores[name] = sc
#         else:
#             scores[method] = score
#     return scores, ""


# def acc_of_seq(choices: List[str], gt: str, pred: str) -> Optional[bool]:
#     if not choices:
#         return None

#     score = difflib.SequenceMatcher(None, pred, gt).quick_ratio()
#     for choice in choices:
#         tmp = difflib.SequenceMatcher(None, pred, str(choice)).quick_ratio()
#         if tmp > score:
#             return False
#     return True


# def expand_letter(text: str, choices: List[str]) -> str:
#     value = str(text).strip()
#     if not choices:
#         return value

#     upper = value.upper()
#     if len(upper) == 1 and upper in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
#         idx = ord(upper) - ord("A")
#         if 0 <= idx < len(choices):
#             return choices[idx]

#     valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
#     parsed = parse_choice_letter(value, valid_letters)
#     if parsed != "FAILED":
#         idx = ord(parsed) - ord("A")
#         if 0 <= idx < len(choices):
#             return choices[idx]
#     return value


# def substring_correct(gt: str, pred: str) -> bool:
#     gt = gt.lower().strip()
#     pred = pred.lower().strip()
#     return bool(gt and pred and (pred in gt or gt in pred))


# def parse_choice_letter(prediction: str, valid_letters: List[str]) -> str:
#     text = str(prediction).strip()
#     if not text:
#         return "FAILED"

#     upper = text.upper()
#     first = upper[0]
#     if first in valid_letters and (len(upper) == 1 or upper[1] in ".:) \n\t"):
#         return first

#     patterns = [
#         r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b",
#         r"\b([A-Z])\s*[\.\):]",
#         r"'([A-Z])'",
#         r'"([A-Z])"',
#     ]
#     for pattern in patterns:
#         matches = [match for match in re.findall(pattern, upper) if match in valid_letters]
#         if len(matches) == 1:
#             return matches[0]

#     matches = [match for match in LETTER_RE.findall(upper) if match in valid_letters]
#     return matches[0] if len(matches) == 1 else "FAILED"


# def choice_letter_accuracy(row: EvalRow) -> Optional[float]:
#     if not row.choices:
#         return None
#     valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(row.choices)])
#     answer_letter = ""
#     norm_gt = normalize_text(row.ground_truth)
#     for letter, choice in zip(valid_letters, row.choices):
#         if normalize_text(choice) == norm_gt:
#             answer_letter = letter
#             break
#     if not answer_letter:
#         return None

#     parsed = parse_choice_letter(row.prediction, valid_letters)
#     norm_pred = normalize_text(row.prediction)
#     return float(parsed == answer_letter or norm_pred == norm_gt or (norm_gt and norm_gt in norm_pred))


# def survival_cindex(rows: List[EvalRow]) -> Optional[float]:
#     event_times = []
#     estimates = []
#     for row in rows:
#         if "survival time" not in row.question.lower():
#             continue
#         pred = row.prediction.strip()
#         gt = row.ground_truth.strip()
#         if not pred.isdecimal() or not gt.isdecimal():
#             continue
#         event_times.append(float(gt))
#         estimates.append(float(pred))

#     if len(estimates) < 2:
#         return None

#     try:
#         from sksurv.metrics import concordance_index_censored
#     except ModuleNotFoundError:
#         return None
#     return float(concordance_index_censored([True] * len(estimates), event_times, estimates, tied_tol=1e-08)[0])


# def evaluate(rows: List[EvalRow], metric_scope: str = "both") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
#     details = []
#     all_gts = {}
#     all_res = {}
#     open_gts = {}
#     open_res = {}
#     total_mcq = correct_mcq = 0
#     total_open = correct_open = 0
#     skipped_no_gt = skipped_no_pred = 0

#     for row in rows:
#         gt = row.ground_truth.strip()
#         pred = row.prediction.strip()
#         if not gt:
#             skipped_no_gt += 1
#         if gt and not pred:
#             skipped_no_pred += 1

#         gt_eval = expand_letter(gt, row.choices)
#         pred_eval = expand_letter(pred, row.choices)
#         is_valid = bool(gt and pred)
#         is_mcq = bool(row.is_mcq or row.choices)

#         exact = float(normalize_text(pred_eval) == normalize_text(gt_eval)) if is_valid else None
#         f1 = token_f1(pred_eval, gt_eval) if is_valid else None
#         seq_acc = None
#         open_substring = None
#         if is_valid:
#             all_gts[row.question_id] = [gt_eval]
#             all_res[row.question_id] = [pred_eval]
#             if is_mcq:
#                 total_mcq += 1
#                 if pred not in {"?", ""}:
#                     seq_acc = acc_of_seq(row.choices, gt_eval, pred_eval)
#                 if seq_acc is None:
#                     seq_acc = False
#                 correct_mcq += int(bool(seq_acc))
#             else:
#                 total_open += 1
#                 open_substring = substring_correct(gt_eval, pred_eval)
#                 correct_open += int(open_substring)
#                 open_gts[row.question_id] = [gt_eval]
#                 open_res[row.question_id] = [pred_eval]

#         letter_acc = choice_letter_accuracy(row) if row.choices and is_valid else None
#         details.append(
#             {
#                 "source_file": row.source_file,
#                 "question_id": row.question_id,
#                 "long_id": row.long_id,
#                 "metadata": row.metadata,
#                 "answer_type": "CLOSED" if is_mcq else "OPEN",
#                 "question": row.question,
#                 "choices_json": json.dumps(row.choices, ensure_ascii=False),
#                 "ground_truth": gt,
#                 "prediction": pred,
#                 "ground_truth_eval": gt_eval,
#                 "prediction_eval": pred_eval,
#                 "normalized_exact_match": exact,
#                 "token_f1": f1,
#                 "mcq_seq_correct": None if seq_acc is None else float(seq_acc),
#                 "mcq_letter_correct": letter_acc,
#                 "open_substring_correct": None if open_substring is None else float(open_substring),
#                 "inference_time_sec": row.inference_time_sec,
#             }
#         )

#     mcq_seq_values = [item["mcq_seq_correct"] for item in details if item["mcq_seq_correct"] is not None]
#     mcq_letter_values = [item["mcq_letter_correct"] for item in details if item["mcq_letter_correct"] is not None]
#     open_substring_values = [
#         item["open_substring_correct"] for item in details if item["open_substring_correct"] is not None
#     ]
#     inference_times = [row.inference_time_sec for row in rows if row.inference_time_sec is not None]
#     open_scores, open_warning = compute_coco_scores(open_gts, open_res)
#     all_scores, all_warning = compute_coco_scores(all_gts, all_res)
#     primary_scores = all_scores if metric_scope == "all" else open_scores
#     metric_scope_note = {
#         "split": "BLEU/METEOR/ROUGE are open-ended only, matching eval_wsivqa.py.",
#         "all": "BLEU/METEOR/ROUGE are computed over every valid sample, matching HJ/WSI-VQA/modules/metrics.py input behavior.",
#         "both": "BLEU/METEOR/ROUGE are open-ended only; all_* keeps the aggregate over every valid sample.",
#     }[metric_scope]

#     summary = {
#         "num_rows": len(rows),
#         "num_valid_for_eval": len(all_gts),
#         "num_valid_for_nlu": len(open_gts),
#         "num_skipped_no_gt": skipped_no_gt,
#         "num_skipped_no_pred": skipped_no_pred,
#         "total_mcq": total_mcq,
#         "correct_mcq": correct_mcq,
#         "mcq_accuracy": correct_mcq / total_mcq if total_mcq else 0.0,
#         "total_open": total_open,
#         "correct_open_substring": correct_open,
#         "open_substring_accuracy": correct_open / total_open if total_open else 0.0,
#         "total_accuracy": (correct_mcq + correct_open) / (total_mcq + total_open) if (total_mcq + total_open) else 0.0,
#         "num_mcq_seq": len(mcq_seq_values),
#         "num_mcq_letter": len(mcq_letter_values),
#         "normalized_exact_match": safe_mean(item["normalized_exact_match"] for item in details),
#         "token_f1": safe_mean(item["token_f1"] for item in details),
#         "mcq_seq_accuracy": safe_mean(mcq_seq_values),
#         "mcq_letter_accuracy": safe_mean(mcq_letter_values),
#         "open_substring_accuracy_detail_mean": safe_mean(open_substring_values),
#         "survival_cindex": survival_cindex(rows),
#         "total_inference_time_sec": sum(inference_times) if inference_times else None,
#         "avg_inference_time_sec": safe_mean(inference_times),
#         "metric_scope": metric_scope,
#         "metric_scope_note": metric_scope_note,
#         "coco_metric_warning": open_warning or all_warning,
#     }
#     summary.update(primary_scores)
#     if metric_scope in {"split", "both"}:
#         summary.update({f"open_{key}": value for key, value in open_scores.items()})
#     if metric_scope in {"all", "both"}:
#         summary.update({f"all_{key}": value for key, value in all_scores.items()})
#     return summary, details


# def load_external_inference_time(path: str) -> Tuple[Optional[float], Optional[int]]:
#     if not path:
#         return None, None
#     payload = read_json(path)
#     seconds = payload.get("inference_time_sec")
#     num_predictions = payload.get("num_predictions")
#     try:
#         seconds = float(seconds) if seconds is not None else None
#     except (TypeError, ValueError):
#         seconds = None
#     try:
#         num_predictions = int(num_predictions) if num_predictions is not None else None
#     except (TypeError, ValueError):
#         num_predictions = None
#     return seconds, num_predictions


# def apply_external_inference_time(summary: Dict[str, Any], seconds: Optional[float], num_predictions: Optional[int]):
#     if seconds is None:
#         return
#     denom = num_predictions or summary.get("num_rows") or 0
#     if summary.get("total_inference_time_sec") is None:
#         summary["total_inference_time_sec"] = seconds
#     if summary.get("avg_inference_time_sec") is None:
#         summary["avg_inference_time_sec"] = seconds / denom if denom else None
#     summary["external_wall_time_sec"] = seconds
#     summary["external_avg_wall_time_sec_per_prediction"] = seconds / denom if denom else None
#     summary["external_inference_num_predictions"] = num_predictions


# def write_json(path: str, payload: Any):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     with open(path, "w", encoding="utf-8") as handle:
#         json.dump(payload, handle, indent=2, ensure_ascii=False)


# def write_csv(path: str, rows: List[Dict[str, Any]]):
#     os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
#     fieldnames = [
#         "source_file",
#         "question_id",
#         "long_id",
#         "metadata",
#         "answer_type",
#         "question",
#         "choices_json",
#         "ground_truth",
#         "prediction",
#         "ground_truth_eval",
#         "prediction_eval",
#         "normalized_exact_match",
#         "token_f1",
#         "mcq_seq_correct",
#         "mcq_letter_correct",
#         "open_substring_correct",
#         "inference_time_sec",
#     ]
#     with open(path, "w", encoding="utf-8", newline="") as handle:
#         writer = csv.DictWriter(handle, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(rows)


# def main():
#     args = parse_args()
#     files = collect_files(args.input, args.glob, args.discover_root)
#     if not files:
#         raise SystemExit("No .json/.jsonl output files found. Pass --input or --glob.")

#     references = load_question_gt(args.questions)
#     rows = []
#     for path in files:
#         rows.extend(load_rows_from_file(path))
#     rows = fill_missing_gt(rows, references)

#     summary, details = evaluate(rows, metric_scope=args.metric_scope)
#     summary["num_files"] = len(files)
#     summary["input_files"] = files

#     external_seconds = args.inference_seconds
#     external_count = None
#     if args.inference_time_json:
#         external_seconds, external_count = load_external_inference_time(args.inference_time_json)
#     apply_external_inference_time(summary, external_seconds, external_count)

#     os.makedirs(args.output_dir, exist_ok=True)
#     summary_path = os.path.join(args.output_dir, f"{args.name}_summary.json")
#     details_path = os.path.join(args.output_dir, f"{args.name}_details.csv")
#     details_json_path = os.path.join(args.output_dir, f"{args.name}_details.json")

#     write_json(summary_path, summary)
#     write_csv(details_path, details)
#     write_json(details_json_path, details)

#     print(f"Evaluated files: {len(files)}")
#     print(
#         f"Rows: {summary['num_rows']}, valid eval rows: {summary['num_valid_for_eval']}, "
#         f"open NLU rows: {summary['num_valid_for_nlu']}"
#     )
#     for key in [
#         "total_accuracy",
#         "mcq_accuracy",
#         "open_substring_accuracy",
#         "BLEU_1",
#         "BLEU_2",
#         "BLEU_3",
#         "BLEU_4",
#         "METEOR",
#         "ROUGE_L",
#         "all_BLEU_1",
#         "all_BLEU_2",
#         "all_BLEU_3",
#         "all_BLEU_4",
#         "all_METEOR",
#         "all_ROUGE_L",
#         "normalized_exact_match",
#         "token_f1",
#         "survival_cindex",
#         "total_inference_time_sec",
#         "avg_inference_time_sec",
#     ]:
#         if key in summary and summary[key] is not None:
#             print(f"{key}: {summary[key]}")
#     if summary.get("coco_metric_warning"):
#         print(f"COCO metrics skipped: {summary['coco_metric_warning']}")
#     print(f"Saved summary JSON: {summary_path}")
#     print(f"Saved details CSV: {details_path}")
#     print(f"Saved details JSON: {details_json_path}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""General WSI-VQA evaluator for model outputs under latentWSI.

This script normalizes common WSI-VQA output formats into rows with:
question_id, question, prediction, ground_truth, choices, inference_time_sec.

It computes the same core NLU metrics used by:
- HJ/PathAgent/eval/metrics_id_only.py
- HJ/WSI-VQA/modules/metrics.py

It also includes general VQA utility metrics that are useful across models:
- PathAgent-style MCQ sequence accuracy
- normalized exact match
- token F1
- optional survival-time C-index when applicable
"""

import argparse
import csv
import difflib
import glob
import hashlib
import json
import math
import os
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


LETTER_RE = re.compile(r"\b([A-Z])\b")
CHOICE_LINE_RE = re.compile(r"^\s*([A-Z])[\.\):]\s*(.+?)\s*$")
INLINE_CHOICE_RE = re.compile(r"(?:^|[;\n]\s*)([A-Z])[\.\):]\s*([^;\n]+)")
CHOICE_MARKER_RE = re.compile(r"(?:^|\s)([A-H])\.\s*")
NUMERIC_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9])\d+(?:\.\d+)?(?!\d)")


@dataclass
class EvalRow:
    source_file: str
    question_id: str
    long_id: str
    question: str
    prediction: str
    ground_truth: str
    choices: List[str]
    is_mcq: bool = False
    metadata: str = ""
    inference_time_sec: Optional[float] = None
    num_output_tokens: Optional[int] = None
    num_answer_tokens: Optional[int] = None
    num_explanation_tokens: Optional[int] = None
    ttft_sec: Optional[float] = None
    flops_total: Optional[float] = None
    flops_prefill: Optional[float] = None
    flops_decode: Optional[float] = None
    flops_reprefill_nocache: Optional[float] = None
    flops_carry_saving: Optional[float] = None
    flops_total_saving: Optional[float] = None


def parse_efficiency(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Pull the additive KV-cache-vs-reprefill fields written by inference.py.

    Reads either the flat keys (ttft_sec, flops_total) or the nested
    `efficiency` block; every field is optional so pre-instrumentation outputs
    parse to all-None and the report simply omits the Efficiency section.
    """
    eff = data.get("efficiency") if isinstance(data.get("efficiency"), dict) else {}

    def num(*keys):
        for src in (data, eff):
            for key in keys:
                value = src.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        return None
        return None

    return {
        "ttft_sec": num("ttft_sec"),
        "flops_total": num("flops_total"),
        "flops_prefill": num("flops_prefill"),
        "flops_decode": num("flops_decode"),
        "flops_reprefill_nocache": num("flops_reprefill_nocache"),
        "flops_carry_saving": num("flops_carry_saving"),
        "flops_total_saving": num("flops_total_saving"),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="General evaluator for WSI-VQA model outputs.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        help="Output files or directories to evaluate. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Glob pattern for output files. Can be passed more than once.",
    )
    parser.add_argument(
        "--discover-root",
        default="",
        help="Scan this root for likely model output JSON/JSONL files only.",
    )
    parser.add_argument(
        "--questions",
        default="",
        help="Optional question/GT JSONL used to fill missing GT by question_id.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval/results",
        help="Directory for metrics JSON/CSV outputs.",
    )
    parser.add_argument(
        "--name",
        default="general_wsi_vqa",
        help="Run name used for output filenames.",
    )
    parser.add_argument(
        "--inference-time-json",
        default="",
        help="Optional JSON file containing inference_time_sec and num_predictions.",
    )
    parser.add_argument(
        "--original-metrics-csv",
        default="",
        help="Optional CSV from the original model evaluator to embed in the summary JSON.",
    )
    parser.add_argument(
        "--inference-seconds",
        type=float,
        default=None,
        help="Optional total inference wall time in seconds.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories from --input. Enabled automatically for directories.",
    )
    parser.add_argument(
        "--metric-scope",
        choices=["split", "all", "both"],
        default="both",
        help=(
            "split: report COCO text metrics on open-ended only, with MCQ accuracy separately. "
            "all: report COCO text metrics on every valid sample, like HJ/WSI-VQA/modules/metrics.py. "
            "both: save both sets; default summary BLEU/METEOR/ROUGE use split/open-ended."
        ),
    )
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def token_prf(prediction: str, answer: str) -> Tuple[float, float, float]:
    """Same tokenization/normalization as token_f1, but returns (f1, precision, recall)."""
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(answer_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(answer_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def metric_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", str(text).lower())


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[idx : idx + n]) for idx in range(0, len(tokens) - n + 1)]


def fallback_bleu(gts: Dict[str, List[str]], res: Dict[str, List[str]], max_n: int = 4) -> List[float]:
    clipped = [0] * max_n
    totals = [0] * max_n
    ref_len = 0
    pred_len = 0

    for key, refs in gts.items():
        ref_tokens = metric_tokenize(refs[0])
        pred_tokens = metric_tokenize(res[key][0])
        ref_len += len(ref_tokens)
        pred_len += len(pred_tokens)
        for n in range(1, max_n + 1):
            pred_counts = Counter(ngrams(pred_tokens, n))
            ref_counts = Counter(ngrams(ref_tokens, n))
            clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            totals[n - 1] += sum(pred_counts.values())

    if pred_len == 0:
        return [0.0] * max_n
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / pred_len)
    scores = []
    for n in range(1, max_n + 1):
        precisions = [(clipped[idx] + 1.0) / (totals[idx] + 1.0) for idx in range(n)]
        scores.append(bp * math.exp(sum(math.log(p) for p in precisions) / n))
    return scores


def lcs_len(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        cur = [0] * (len(b) + 1)
        for idx, token_b in enumerate(b, 1):
            cur[idx] = prev[idx - 1] + 1 if token_a == token_b else max(prev[idx], cur[idx - 1])
        prev = cur
    return prev[-1]


def fallback_rouge_l(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> float:
    scores = []
    for key, refs in gts.items():
        ref_tokens = metric_tokenize(refs[0])
        pred_tokens = metric_tokenize(res[key][0])
        if not ref_tokens or not pred_tokens:
            scores.append(0.0)
            continue
        lcs = lcs_len(ref_tokens, pred_tokens)
        precision = lcs / len(pred_tokens)
        recall = lcs / len(ref_tokens)
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores) if scores else 0.0


def fallback_coco_scores(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
    per_sample_bleu = [
        fallback_bleu({key: gts[key]}, {key: res[key]})
        for key in gts
    ]
    bleu = [
        safe_mean(sample_scores[index] for sample_scores in per_sample_bleu) or 0.0
        for index in range(4)
    ]
    return {
        "BLEU_1": bleu[0],
        "BLEU_2": bleu[1],
        "BLEU_3": bleu[2],
        "BLEU_4": bleu[3],
        "METEOR": 0.0,
        "ROUGE_L": fallback_rouge_l(gts, res),
    }


def compute_pathagent_scores(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
    """Match PathAgent's official corpus-level BLEU, METEOR, and ROUGE calculation."""
    if not gts:
        return {
            "BLEU_1": 0.0,
            "BLEU_2": 0.0,
            "BLEU_3": 0.0,
            "BLEU_4": 0.0,
            "METEOR": 0.0,
            "ROUGE_L": 0.0,
        }
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    scores: Dict[str, float] = {}
    for scorer, names in (
        (Bleu(4), ("BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4")),
        (Meteor(), ("METEOR",)),
        (Rouge(), ("ROUGE_L",)),
    ):
        score, _ = scorer.compute_score(gts, res)
        if len(names) == 1:
            scores[names[0]] = float(score)
        else:
            for name, value in zip(names, score, strict=True):
                scores[name] = float(value)
    return scores


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [value for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return None
    return sum(clean) / len(clean)


def make_unique_id(long_id: str, question_text: str) -> str:
    q_hash = hashlib.md5(question_text.encode("utf-8")).hexdigest()[:8]
    return f"{long_id}_{q_hash}"


def first_present(data: Dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def as_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if value is None:
        return ""
    return str(value)


def parse_question_choices(question: Any) -> List[str]:
    choices = []
    text = as_text(question)
    for line in text.splitlines():
        match = CHOICE_LINE_RE.match(line)
        if match:
            choices.append(match.group(2).strip())
    if choices:
        return choices

    for _, choice in INLINE_CHOICE_RE.findall(text):
        choice = choice.strip()
        if choice.lower().startswith("answer with"):
            continue
        choices.append(choice)
    return choices


def parse_marker_choices(value: Any) -> List[str]:
    text = as_text(value)
    if not text.strip():
        return []
    matches = list(CHOICE_MARKER_RE.finditer(text))
    if not matches:
        return []
    choices = []
    for idx, match in enumerate(matches):
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        choice = text[match.end() : end].strip()
        choice = re.sub(r"\n?Answer with the best choice\.?\s*$", "", choice, flags=re.I).strip()
        if choice:
            choices.append(choice)
    return choices


def coerce_choices(data: Dict[str, Any]) -> List[str]:
    raw = first_present(data, ["choices", "Choice", "choices_json"], default=[])
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return parse_question_choices(first_present(data, ["question", "Question"], default=""))
        try:
            parsed = json.loads(raw)
            raw = parsed
        except json.JSONDecodeError:
            parsed_choices = parse_marker_choices(raw)
            if parsed_choices:
                return parsed_choices
            raw = [raw]
    if raw is None:
        return parse_question_choices(first_present(data, ["question", "Question"], default=""))
    if isinstance(raw, dict):
        return [str(raw[key]).strip() for key in sorted(raw) if str(raw[key]).strip()]
    if isinstance(raw, list):
        choices = [as_text(item).strip() for item in raw if as_text(item).strip()]
        return choices or parse_question_choices(first_present(data, ["question", "Question"], default=""))
    return [str(raw).strip()] if str(raw).strip() else []


def infer_is_mcq(data: Dict[str, Any], choices: List[str]) -> bool:
    if choices:
        return True
    value = data.get("is_mcq")
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "closed", "mcq"}:
        return True
    answer_type = as_text(first_present(data, ["answer_type", "metadata", "type"], default="")).strip().lower()
    return answer_type in {"closed", "close-ended", "closed-ended", "multiple-choice", "multiple_choice", "mcq"}


def parse_inference_time(data: Dict[str, Any]) -> Optional[float]:
    value = first_present(
        data,
        ["inference_time_sec", "inference_time", "elapsed_sec", "time_sec", "latency_sec"],
        default=None,
    )
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_output_tokens(data: Dict[str, Any]) -> Optional[int]:
    value = first_present(
        data,
        ["num_output_tokens", "output_tokens", "completion_tokens", "generated_tokens", "Token", "token"],
        default=None,
    )
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_named_tokens(data: Dict[str, Any], keys: List[str]) -> Optional[int]:
    value = first_present(data, keys, default=None)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_from_dict(data: Dict[str, Any], source_file: str, fallback_id: str) -> Optional[EvalRow]:
    question = as_text(first_present(data, ["raw_question", "question", "Question", "prompt"], default=""))
    if "res" in data:
        prediction_value = first_present(data, ["res"], default="")
    elif "T-answer" in data or "ground_truth" in data or "gt" in data:
        prediction_value = first_present(
            data,
            ["pred_answer", "prediction", "Output", "output", "answer", "text", "response", "res"],
            default="",
        )
    else:
        prediction_value = first_present(
            data,
            ["pred_answer", "prediction", "Output", "output", "answer", "text", "response", "res"],
            default="",
        )
    prediction = as_text(prediction_value).strip()
    if prediction == question and "text" in data:
        prediction = ""
    ground_truth = as_text(
        first_present(data, ["ground_truth", "T-answer", "gts", "gt", "answer_gt", "target", "label", "Answer"], default="")
    ).strip()
    question_id = as_text(first_present(data, ["question_id", "id", "Id"], default="")).strip()
    long_id = as_text(first_present(data, ["long_id", "wsi_full_id", "slide_id", "case_id", "image", "Id"], default="")).strip()
    metadata = as_text(first_present(data, ["metadata", "answer_type"], default="")).strip()
    choices = coerce_choices(data)
    is_mcq = infer_is_mcq(data, choices)

    if not question_id:
        question_id = make_unique_id(long_id, question) if long_id and question else fallback_id

    if not prediction and not ground_truth:
        return None

    return EvalRow(
        source_file=source_file,
        question_id=question_id,
        long_id=long_id,
        question=question,
        prediction=prediction,
        ground_truth=ground_truth,
        choices=choices,
        is_mcq=is_mcq,
        metadata=metadata,
        inference_time_sec=parse_inference_time(data),
        num_output_tokens=parse_output_tokens(data),
        num_answer_tokens=parse_named_tokens(data, ["num_answer_tokens", "answer_tokens"]),
        num_explanation_tokens=parse_named_tokens(data, ["num_explanation_tokens", "explanation_tokens"]),
        **parse_efficiency(data),
    )


def rows_from_json_payload(payload: Any, source_file: str) -> List[EvalRow]:
    if isinstance(payload, list):
        rows = []
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                row = row_from_dict(item, source_file, f"{os.path.basename(source_file)}_{idx}")
                if row:
                    rows.append(row)
        return rows

    if isinstance(payload, dict):
        if all(isinstance(value, list) for value in payload.values()):
            rows = []
            for case_id, samples in payload.items():
                for idx, item in enumerate(samples):
                    if isinstance(item, dict):
                        merged = dict(item)
                        merged.setdefault("case_id", case_id)
                        row = row_from_dict(merged, source_file, f"{case_id}_{idx}")
                        if row:
                            rows.append(row)
            return rows

        row = row_from_dict(payload, source_file, os.path.splitext(os.path.basename(source_file))[0])
        return [row] if row else []

    return []


def load_rows_from_file(path: str) -> List[EvalRow]:
    if path.endswith(".jsonl"):
        rows = []
        for idx, item in enumerate(read_jsonl(path)):
            row = row_from_dict(item, path, f"{os.path.basename(path)}_{idx}")
            if row:
                rows.append(row)
        return rows

    if path.endswith(".json"):
        return rows_from_json_payload(read_json(path), path)

    return []


def is_likely_output_file(path: str) -> bool:
    lower = path.lower()
    include_tokens = ["output", "outputs", "result", "results", "answer", "answers", "pred", "prediction", "vis"]
    exclude_tokens = [
        "/data/",
        "/playground/data/",
        "/checkpoints/",
        "/checkpoint/",
        "/models/",
        "/model/",
        "/cache/",
        "__pycache__",
        "/patches_output/",
    ]
    return any(token in lower for token in include_tokens) and not any(token in lower for token in exclude_tokens)


def discover_output_files(root: str) -> List[str]:
    if not root:
        return []
    files = []
    for pattern in ["**/*.json", "**/*.jsonl"]:
        for path in glob.glob(os.path.join(root, pattern), recursive=True):
            if os.path.isfile(path) and is_likely_output_file(path):
                files.append(path)
    return files


def collect_files(inputs: List[str], patterns: List[str], discover_root: str = "") -> List[str]:
    files = []
    for item in inputs:
        if os.path.isdir(item):
            files.extend(
                path
                for path in glob.glob(os.path.join(item, "**", "*"), recursive=True)
                if path.endswith((".json", ".jsonl"))
            )
        elif os.path.isfile(item) and item.endswith((".json", ".jsonl")):
            files.append(item)

    for pattern in patterns:
        files.extend(path for path in glob.glob(pattern, recursive=True) if path.endswith((".json", ".jsonl")))

    files.extend(discover_output_files(discover_root))

    unique = []
    seen = set()
    for path in files:
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            unique.append(path)
    return sorted(unique)


def load_question_gt(path: str) -> Dict[str, Dict[Any, EvalRow]]:
    if not path:
        return {}
    rows = load_rows_from_file(path)
    by_id = {row.question_id: row for row in rows if row.question_id}
    by_pair = {
        (row.long_id, row.question): row
        for row in rows
        if row.long_id and row.question
    }
    return {"by_id": by_id, "by_pair": by_pair}


def fill_missing_gt(rows: List[EvalRow], references: Dict[str, Dict[Any, EvalRow]]) -> List[EvalRow]:
    if not references:
        return rows

    filled = []
    for row in rows:
        ref = references.get("by_id", {}).get(row.question_id)
        if ref is None and row.long_id and row.question:
            ref = references.get("by_pair", {}).get((row.long_id, row.question))
        if ref and not row.ground_truth:
            row.ground_truth = ref.ground_truth
        if ref and not row.question:
            row.question = ref.question
        if ref and not row.choices:
            row.choices = ref.choices
        if ref and not row.is_mcq:
            row.is_mcq = ref.is_mcq
        filled.append(row)
    return filled


def _run_scorer_with_timeout(scorer, gts, res, timeout=30):
    """Run scorer.compute_score in a thread with timeout. Returns (score, scores_list) or raises TimeoutError."""
    import threading
    result = [None]
    exc_box = [None]

    def _run():
        try:
            result[0] = scorer.compute_score(gts, res)
        except Exception as e:
            exc_box[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        raise TimeoutError(f"scorer timed out after {timeout}s (likely METEOR Java hang)")
    if exc_box[0] is not None:
        raise exc_box[0]
    return result[0]


def compute_coco_scores(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Tuple[Dict[str, float], str]:
    empty_scores = {
        "BLEU_1": 0.0,
        "BLEU_2": 0.0,
        "BLEU_3": 0.0,
        "BLEU_4": 0.0,
        "METEOR": 0.0,
        "ROUGE_L": 0.0,
    }
    if not gts:
        return empty_scores, ""
    # Normalize (lowercase + strip punctuation) so BLEU/ROUGE are case/punct
    # insensitive, matching normalized_exact_match/token_f1. Without this,
    # GT "Positive" vs pred "positive" counts as a mismatch and unigram
    # overlap collapses (e.g. open BLEU-1 0.0079 -> 0.15). METEOR already
    # lowercases/stems internally, so it is unaffected. Applied BEFORE the
    # try/except so both the pycocoevalcap path and the fallback_coco_scores
    # path receive the same normalized captions (the fallback tokenizer already
    # lowercases, so this also removes the case-sensitive vs -insensitive
    # divergence between the two paths). Placeholder token avoids feeding an
    # empty hypothesis/reference to the scorers.
    def _norm_caption(value: str) -> str:
        return normalize_text(value) or "<empty>"

    gts = {key: [_norm_caption(v) for v in vals] for key, vals in gts.items()}
    res = {key: [_norm_caption(v) for v in vals] for key, vals in res.items()}
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ModuleNotFoundError as exc:
        return fallback_coco_scores(gts, res), f"pycocoevalcap unavailable; using fallback BLEU/ROUGE and METEOR=0: {exc}"

    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
    ]
    scores = {}
    warning_parts = []
    for scorer, method in scorers:
        try:
            _score, per_sample_scores = _run_scorer_with_timeout(
                scorer, gts, res, timeout=30
            )
            if isinstance(method, list):
                for sample_scores, name in zip(per_sample_scores, method):
                    scores[name] = safe_mean(sample_scores) or 0.0
            else:
                scores[method] = safe_mean(per_sample_scores) or 0.0
        except Exception as e:
            name_str = method if isinstance(method, str) else "/".join(method)
            warning_parts.append(f"{name_str} failed: {e}")
            if isinstance(method, list):
                for name in method:
                    scores[name] = 0.0
            else:
                scores[method] = 0.0
    for key in ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L"]:
        if key not in scores:
            scores[key] = 0.0
    warning = "; ".join(warning_parts)
    return scores, warning


def acc_of_seq(choices: List[str], gt: str, pred: str) -> Optional[bool]:
    if not choices:
        return None

    score = difflib.SequenceMatcher(None, pred, gt).quick_ratio()
    for choice in choices:
        tmp = difflib.SequenceMatcher(None, pred, str(choice)).quick_ratio()
        if tmp > score:
            return False
    return True


def expand_letter(text: str, choices: List[str]) -> str:
    value = str(text).strip()
    if not choices:
        return value

    upper = value.upper()
    if len(upper) == 1 and upper in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        idx = ord(upper) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]

    valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)])
    parsed = parse_choice_letter(value, valid_letters)
    if parsed != "FAILED":
        idx = ord(parsed) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return value


def substring_correct(gt: str, pred: str) -> bool:
    gt = gt.lower().strip()
    pred = pred.lower().strip()
    if not gt or not pred:
        return False

    # Text answers remain substring-tolerant, but numeric answers must match
    # whole numeric tokens: ``125`` must not match the shorter ``12``.
    normalized_gt = re.sub(r"(?<=\d),(?=\d)", "", gt)
    normalized_pred = re.sub(r"(?<=\d),(?=\d)", "", pred)
    gt_numbers = set(NUMERIC_TOKEN_RE.findall(normalized_gt))
    pred_numbers = set(NUMERIC_TOKEN_RE.findall(normalized_pred))
    if gt_numbers or pred_numbers:
        if gt_numbers != pred_numbers:
            return False
        gt = normalized_gt
        pred = normalized_pred
    return pred in gt or gt in pred


def parse_choice_letter(prediction: str, valid_letters: List[str]) -> str:
    text = str(prediction).strip()
    if not text:
        return "FAILED"

    upper = text.upper()
    first = upper[0]
    if first in valid_letters and (len(upper) == 1 or upper[1] in ".:) \n\t"):
        return first

    patterns = [
        r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b",
        r"\b([A-Z])\s*[\.\):]",
        r"'([A-Z])'",
        r'"([A-Z])"',
    ]
    for pattern in patterns:
        matches = [match for match in re.findall(pattern, upper) if match in valid_letters]
        if len(matches) == 1:
            return matches[0]

    matches = [match for match in LETTER_RE.findall(upper) if match in valid_letters]
    return matches[0] if len(matches) == 1 else "FAILED"


def choice_letter_accuracy(row: EvalRow) -> Optional[float]:
    if not row.choices:
        return None
    valid_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(row.choices)])
    answer_letter = ""
    norm_gt = normalize_text(row.ground_truth)
    for letter, choice in zip(valid_letters, row.choices):
        if normalize_text(choice) == norm_gt:
            answer_letter = letter
            break
    if not answer_letter:
        return None

    parsed = parse_choice_letter(row.prediction, valid_letters)
    norm_pred = normalize_text(row.prediction)
    return float(parsed == answer_letter or norm_pred == norm_gt or (norm_gt and norm_gt in norm_pred))


def survival_cindex(rows: List[EvalRow]) -> Optional[float]:
    event_times = []
    estimates = []
    for row in rows:
        if "survival time" not in row.question.lower():
            continue
        pred = row.prediction.strip()
        gt = row.ground_truth.strip()
        if not pred.isdecimal() or not gt.isdecimal():
            continue
        event_times.append(float(gt))
        estimates.append(float(pred))

    if len(estimates) < 2:
        return None

    try:
        from sksurv.metrics import concordance_index_censored
    except ModuleNotFoundError:
        return None
    return float(concordance_index_censored([True] * len(estimates), event_times, estimates, tied_tol=1e-08)[0])


def evaluate(
    rows: List[EvalRow],
    metric_scope: str = "both",
    include_text_metrics: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    details = []
    all_gts = {}
    all_res = {}
    open_gts = {}
    open_res = {}
    total_mcq = correct_mcq = 0
    total_open = correct_open = 0
    skipped_no_gt = skipped_no_pred = 0

    for row in rows:
        gt = row.ground_truth.strip()
        pred = row.prediction.strip()
        if not gt:
            skipped_no_gt += 1
        if gt and not pred:
            skipped_no_pred += 1

        gt_eval = expand_letter(gt, row.choices)
        pred_eval = expand_letter(pred, row.choices)
        is_valid = bool(gt and pred)
        is_mcq = bool(row.is_mcq or row.choices)

        exact = float(normalize_text(pred_eval) == normalize_text(gt_eval)) if is_valid else None
        if is_valid:
            f1, prec, rec = token_prf(pred_eval, gt_eval)
        else:
            f1 = prec = rec = None
        seq_acc = None
        open_substring = None
        if is_valid:
            all_gts[row.question_id] = [gt_eval]
            all_res[row.question_id] = [pred_eval]
            if is_mcq:
                total_mcq += 1
                if pred not in {"?", ""}:
                    seq_acc = acc_of_seq(row.choices, gt_eval, pred_eval)
                if seq_acc is None:
                    seq_acc = False
                correct_mcq += int(bool(seq_acc))
            else:
                total_open += 1
                open_substring = substring_correct(gt_eval, pred_eval)
                correct_open += int(open_substring)
                open_gts[row.question_id] = [gt_eval]
                open_res[row.question_id] = [pred_eval]

        letter_acc = choice_letter_accuracy(row) if row.choices and is_valid else None
        details.append(
            {
                "source_file": row.source_file,
                "question_id": row.question_id,
                "long_id": row.long_id,
                "metadata": row.metadata,
                "answer_type": "CLOSED" if is_mcq else "OPEN",
                "question": row.question,
                "choices_json": json.dumps(row.choices, ensure_ascii=False),
                "ground_truth": gt,
                "prediction": pred,
                "ground_truth_eval": gt_eval,
                "prediction_eval": pred_eval,
                "normalized_exact_match": exact,
                "token_f1": f1,
                "token_precision": prec,
                "token_recall": rec,
                "mcq_seq_correct": None if seq_acc is None else float(seq_acc),
                "mcq_letter_correct": letter_acc,
                "open_substring_correct": None if open_substring is None else float(open_substring),
                "inference_time_sec": row.inference_time_sec,
                "num_output_tokens": row.num_output_tokens,
                "num_answer_tokens": row.num_answer_tokens,
                "num_explanation_tokens": row.num_explanation_tokens,
            }
        )

    mcq_seq_values = [item["mcq_seq_correct"] for item in details if item["mcq_seq_correct"] is not None]
    mcq_letter_values = [item["mcq_letter_correct"] for item in details if item["mcq_letter_correct"] is not None]
    open_substring_values = [
        item["open_substring_correct"] for item in details if item["open_substring_correct"] is not None
    ]
    open_details = [item for item in details if item["answer_type"] == "OPEN"]
    inference_times = [row.inference_time_sec for row in rows if row.inference_time_sec is not None]
    ttft_values = [row.ttft_sec for row in rows if row.ttft_sec is not None]
    flops_total_values = [row.flops_total for row in rows if row.flops_total is not None]
    flops_prefill_values = [row.flops_prefill for row in rows if row.flops_prefill is not None]
    flops_decode_values = [row.flops_decode for row in rows if row.flops_decode is not None]
    flops_nocache_values = [row.flops_reprefill_nocache for row in rows if row.flops_reprefill_nocache is not None]
    flops_carry_saving_values = [row.flops_carry_saving for row in rows if row.flops_carry_saving is not None]
    flops_total_saving_values = [row.flops_total_saving for row in rows if row.flops_total_saving is not None]
    output_tokens = [row.num_output_tokens for row in rows if row.num_output_tokens is not None]
    answer_tokens = [row.num_answer_tokens for row in rows if row.num_answer_tokens is not None]
    explanation_tokens = [row.num_explanation_tokens for row in rows if row.num_explanation_tokens is not None]
    if include_text_metrics:
        open_scores, open_warning = compute_coco_scores(open_gts, open_res)
        all_scores, all_warning = compute_coco_scores(all_gts, all_res)
    else:
        empty_text_scores = {
            "BLEU_1": None,
            "BLEU_2": None,
            "BLEU_3": None,
            "BLEU_4": None,
            "METEOR": None,
            "ROUGE_L": None,
        }
        open_scores = empty_text_scores
        all_scores = empty_text_scores
        open_warning = ""
        all_warning = ""
    primary_scores = all_scores if metric_scope == "all" else open_scores
    metric_scope_note = {
        "split": "BLEU/METEOR/ROUGE are sample-wise macro averages over open-ended answers.",
        "all": "BLEU/METEOR/ROUGE are sample-wise macro averages over every valid sample.",
        "both": "BLEU/METEOR/ROUGE are sample-wise macro averages over open-ended answers; all_* uses every valid sample.",
    }[metric_scope]

    summary = {
        "num_rows": len(rows),
        "num_valid_for_eval": len(all_gts),
        "num_valid_for_nlu": len(open_gts),
        "num_skipped_no_gt": skipped_no_gt,
        "num_skipped_no_pred": skipped_no_pred,
        "total_mcq": total_mcq,
        "correct_mcq": correct_mcq,
        "mcq_accuracy": correct_mcq / total_mcq if total_mcq else 0.0,
        "total_open": total_open,
        "correct_open_substring": correct_open,
        "open_substring_accuracy": correct_open / total_open if total_open else 0.0,
        "total_accuracy": (correct_mcq + correct_open) / (total_mcq + total_open) if (total_mcq + total_open) else 0.0,
        "num_mcq_seq": len(mcq_seq_values),
        "num_mcq_letter": len(mcq_letter_values),
        "normalized_exact_match": safe_mean(item["normalized_exact_match"] for item in details),
        "token_f1": safe_mean(item["token_f1"] for item in details),
        "token_precision": safe_mean(item["token_precision"] for item in details),
        "token_recall": safe_mean(item["token_recall"] for item in details),
        "open_token_f1": safe_mean(item["token_f1"] for item in open_details),
        "open_token_precision": safe_mean(item["token_precision"] for item in open_details),
        "open_token_recall": safe_mean(item["token_recall"] for item in open_details),
        "mcq_seq_accuracy": safe_mean(mcq_seq_values),
        "mcq_letter_accuracy": safe_mean(mcq_letter_values),
        "open_substring_accuracy_detail_mean": safe_mean(open_substring_values),
        "survival_cindex": survival_cindex(rows),
        "total_inference_time_sec": sum(inference_times) if inference_times else None,
        "avg_inference_time_sec": safe_mean(inference_times),
        "num_ttft": len(ttft_values),
        "avg_ttft_sec": safe_mean(ttft_values),
        "num_flops": len(flops_total_values),
        "avg_flops_total": safe_mean(flops_total_values),
        "avg_flops_prefill": safe_mean(flops_prefill_values),
        "avg_flops_decode": safe_mean(flops_decode_values),
        "total_flops": sum(flops_total_values) if flops_total_values else None,
        "avg_flops_reprefill_nocache": safe_mean(flops_nocache_values),
        "avg_flops_carry_saving": safe_mean(flops_carry_saving_values),
        "avg_flops_total_saving": safe_mean(flops_total_saving_values),
        "carry_saving_ratio": (
            (sum(flops_carry_saving_values) / (sum(flops_prefill_values) + sum(flops_carry_saving_values)))
            if flops_carry_saving_values and flops_prefill_values else None
        ),
        "total_saving_ratio": (
            (sum(flops_total_saving_values) / sum(flops_nocache_values))
            if flops_total_saving_values and flops_nocache_values else None
        ),
        "total_output_tokens": sum(output_tokens) if output_tokens else None,
        "avg_output_tokens_per_question": safe_mean(output_tokens),
        "total_answer_tokens": sum(answer_tokens) if answer_tokens else None,
        "avg_answer_tokens_per_question": safe_mean(answer_tokens),
        "total_explanation_tokens": sum(explanation_tokens) if explanation_tokens else None,
        "avg_explanation_tokens_per_question": safe_mean(explanation_tokens),
        "metric_scope": metric_scope,
        "metric_scope_note": metric_scope_note,
        "coco_metric_warning": open_warning or all_warning,
    }
    summary.update(primary_scores)
    if metric_scope in {"split", "both"}:
        summary.update({f"open_{key}": value for key, value in open_scores.items()})
    if metric_scope in {"all", "both"}:
        summary.update({f"all_{key}": value for key, value in all_scores.items()})
    return summary, details


def load_external_inference_time(path: str) -> Tuple[Optional[float], Optional[int]]:
    if not path:
        return None, None
    payload = read_json(path)
    seconds = payload.get("inference_time_sec")
    num_predictions = payload.get("num_predictions")
    try:
        seconds = float(seconds) if seconds is not None else None
    except (TypeError, ValueError):
        seconds = None
    try:
        num_predictions = int(num_predictions) if num_predictions is not None else None
    except (TypeError, ValueError):
        num_predictions = None
    return seconds, num_predictions


def load_original_metrics_csv(path: str) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Original metrics CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        return {"source_csv": path, "rows": []}

    def coerce_value(value: Any) -> Any:
        if value is None:
            return None
        text = str(value).strip()
        if text == "":
            return text
        try:
            if re.fullmatch(r"[-+]?\d+", text):
                return int(text)
            if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text):
                return float(text)
        except ValueError:
            pass
        return value

    coerced_rows = [{key: coerce_value(value) for key, value in row.items()} for row in rows]
    return {
        "source_csv": path,
        "row": coerced_rows[0] if len(coerced_rows) == 1 else None,
        "rows": coerced_rows,
    }


def apply_external_inference_time(summary: Dict[str, Any], seconds: Optional[float], num_predictions: Optional[int]):
    if seconds is None:
        return
    denom = num_predictions or summary.get("num_rows") or 0
    if summary.get("total_inference_time_sec") is None:
        summary["total_inference_time_sec"] = seconds
    if summary.get("avg_inference_time_sec") is None:
        summary["avg_inference_time_sec"] = seconds / denom if denom else None
    summary["external_wall_time_sec"] = seconds
    summary["external_avg_wall_time_sec_per_prediction"] = seconds / denom if denom else None
    summary["external_inference_num_predictions"] = num_predictions


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def build_readable_report(summary: Dict[str, Any]) -> str:
    lines = [
        "General WSI-VQA Evaluation Report",
        "=" * 34,
        "",
        "[Dataset]",
        f"- files: {summary.get('num_files', 0)}",
        f"- rows loaded: {summary.get('num_rows', 0)}",
        f"- valid eval rows: {summary.get('num_valid_for_eval', 0)}",
        f"- skipped without GT: {summary.get('num_skipped_no_gt', 0)}",
        f"- skipped without prediction: {summary.get('num_skipped_no_pred', 0)}",
        "",
        "[Answer-Type Split]",
        f"- MCQ/CLOSED: {summary.get('correct_mcq', 0)}/{summary.get('total_mcq', 0)} = {fmt_pct(summary.get('mcq_accuracy'))}",
        f"- OPEN: {summary.get('correct_open_substring', 0)}/{summary.get('total_open', 0)} = {fmt_pct(summary.get('open_substring_accuracy'))}",
        f"- TOTAL: {fmt_pct(summary.get('total_accuracy'))}",
        "",
        "[MCQ Metrics]",
        f"- mcq_accuracy: {fmt_pct(summary.get('mcq_accuracy'))}",
        f"- mcq_seq_accuracy: {fmt_pct(summary.get('mcq_seq_accuracy'))}",
        f"- mcq_letter_accuracy: {fmt_pct(summary.get('mcq_letter_accuracy'))}",
        "",
        "[Open-Ended Metrics]",
        f"- open_substring_accuracy: {fmt_pct(summary.get('open_substring_accuracy'))}",
        f"- open_token_f1: {fmt_pct(summary.get('open_token_f1'))}",
        f"- open_token_precision: {fmt_pct(summary.get('open_token_precision'))}",
        f"- open_token_recall: {fmt_pct(summary.get('open_token_recall'))}",
        f"- open_BLEU_1: {fmt_num(summary.get('open_BLEU_1'))}",
        f"- open_BLEU_2: {fmt_num(summary.get('open_BLEU_2'))}",
        f"- open_BLEU_3: {fmt_num(summary.get('open_BLEU_3'))}",
        f"- open_BLEU_4: {fmt_num(summary.get('open_BLEU_4'))}",
        f"- open_METEOR: {fmt_num(summary.get('open_METEOR'))}",
        f"- open_ROUGE_L: {fmt_num(summary.get('open_ROUGE_L'))}",
        "  meaning: open_* metrics include OPEN rows only, excluding MCQ/CLOSED rows.",
        "",
        "[All Valid Rows Metrics]",
        f"- normalized_exact_match: {fmt_pct(summary.get('normalized_exact_match'))}",
        f"- token_f1: {fmt_pct(summary.get('token_f1'))}",
        f"- token_precision: {fmt_pct(summary.get('token_precision'))}",
        f"- token_recall: {fmt_pct(summary.get('token_recall'))}",
        f"- all_BLEU_1: {fmt_num(summary.get('all_BLEU_1'))}",
        f"- all_BLEU_2: {fmt_num(summary.get('all_BLEU_2'))}",
        f"- all_BLEU_3: {fmt_num(summary.get('all_BLEU_3'))}",
        f"- all_BLEU_4: {fmt_num(summary.get('all_BLEU_4'))}",
        f"- all_METEOR: {fmt_num(summary.get('all_METEOR'))}",
        f"- all_ROUGE_L: {fmt_num(summary.get('all_ROUGE_L'))}",
        "  meaning: all_* metrics include every valid row, both MCQ/CLOSED and OPEN.",
        "",
        "[Primary COCO Scope]",
        f"- metric_scope: {summary.get('metric_scope')}",
        f"- primary BLEU_1: {fmt_num(summary.get('BLEU_1'))}",
        f"- primary BLEU_4: {fmt_num(summary.get('BLEU_4'))}",
        f"- primary METEOR: {fmt_num(summary.get('METEOR'))}",
        f"- primary ROUGE_L: {fmt_num(summary.get('ROUGE_L'))}",
        "",
        "[Inference Time]",
        f"- total_inference_time_sec: {fmt_num(summary.get('total_inference_time_sec'), 2)}",
        f"- avg_inference_time_sec: {fmt_num(summary.get('avg_inference_time_sec'), 2)}",
        "",
        "[Token]",
        f"- total_output_tokens: {fmt_num(summary.get('total_output_tokens'), 0)}",
        f"- avg_output_tokens_per_question: {fmt_num(summary.get('avg_output_tokens_per_question'), 2)}",
        f"- total_answer_tokens: {fmt_num(summary.get('total_answer_tokens'), 0)}",
        f"- avg_answer_tokens_per_question: {fmt_num(summary.get('avg_answer_tokens_per_question'), 2)}",
        f"- total_explanation_tokens: {fmt_num(summary.get('total_explanation_tokens'), 0)}",
        f"- avg_explanation_tokens_per_question: {fmt_num(summary.get('avg_explanation_tokens_per_question'), 2)}",
    ]

    if summary.get("external_wall_time_sec") is not None:
        lines.extend([
            f"- external_wall_time_sec: {fmt_num(summary.get('external_wall_time_sec'), 2)}",
            f"- external_avg_wall_time_sec_per_prediction: {fmt_num(summary.get('external_avg_wall_time_sec_per_prediction'), 2)}",
        ])

    # Efficiency (KV-cache carry vs re-prefill): only shown when the run was
    # instrumented (inference.py wrote ttft_sec / flops_*). Absent otherwise.
    if summary.get("num_ttft") or summary.get("num_flops"):
        gflops = lambda v: fmt_num(v / 1e9, 2) if v is not None else "N/A"
        lines.extend([
            "",
            "[Efficiency: KV-cache carry vs re-prefill]",
            f"- avg_ttft_sec: {fmt_num(summary.get('avg_ttft_sec'), 4)}  (n={summary.get('num_ttft', 0)}; final-answer time-to-first-token)",
            f"- avg_flops_total_GFLOPs: {gflops(summary.get('avg_flops_total'))}  (n={summary.get('num_flops', 0)}; per item, all LM forwards)",
            f"- avg_flops_prefill_GFLOPs: {gflops(summary.get('avg_flops_prefill'))}",
            f"- avg_flops_decode_GFLOPs: {gflops(summary.get('avg_flops_decode'))}",
            f"- total_flops_GFLOPs: {gflops(summary.get('total_flops'))}",
            "  meaning: analytical leading-order LM FLOPs; lower total = less re-prefill.",
            "  compare the same number across --kv_carry on/off runs for the KV-cache saving.",
        ])
        pct = lambda v: f"{v*100:.1f}%" if v is not None else "N/A"
        cs = summary.get("avg_flops_carry_saving")
        if cs is not None or summary.get("avg_flops_reprefill_nocache") is not None:
            lines.extend([
                "",
                "  -- counterfactual (within-run, trajectory-independent) --",
                f"- avg_flops_reprefill_nocache_GFLOPs: {gflops(summary.get('avg_flops_reprefill_nocache'))}  (if every forward re-prefilled its full context)",
                f"- avg_flops_carry_saving_GFLOPs: {gflops(cs)}  (agentic KV-carry: append reuse vs re-prefill)",
                f"- carry_saving_ratio: {pct(summary.get('carry_saving_ratio'))}  (novelty metric; prefill/append scope)",
                f"- avg_flops_total_saving_GFLOPs: {gflops(summary.get('avg_flops_total_saving'))}  (incl. standard decode KV cache)",
                f"- total_saving_ratio: {pct(summary.get('total_saving_ratio'))}",
                "  meaning: actual vs a no-cache run that recomputes each forward's whole context.",
                "  carry_saving isolates the KV-carry contribution independent of which path the run took.",
            ])

    if summary.get("survival_cindex") is not None:
        lines.extend(["", "[Survival]", f"- survival_cindex: {fmt_num(summary.get('survival_cindex'))}"])

    if summary.get("coco_metric_warning"):
        lines.extend(["", "[Warning]", f"- {summary['coco_metric_warning']}"])

    return "\n".join(lines) + "\n"


def print_readable_summary(summary: Dict[str, Any], report_path: str, details_path: str, details_json_path: str):
    print("\n=== General WSI-VQA Evaluation ===")
    print(
        f"Files: {summary.get('num_files', 0)} | Rows: {summary['num_rows']} | "
        f"Valid: {summary['num_valid_for_eval']} | "
        f"Skipped(no GT/pred): {summary['num_skipped_no_gt']}/{summary['num_skipped_no_pred']}"
    )

    print("\n[Answer-Type Split]")
    print(f"MCQ/CLOSED: {summary['correct_mcq']}/{summary['total_mcq']} = {fmt_pct(summary['mcq_accuracy'])}")
    print(
        f"OPEN:       {summary['correct_open_substring']}/{summary['total_open']} = "
        f"{fmt_pct(summary['open_substring_accuracy'])}  (substring match)"
    )
    print(f"TOTAL:      {fmt_pct(summary['total_accuracy'])}  (MCQ correct + OPEN substring correct)")

    print("\n[Open-Ended Text Metrics: OPEN only]")
    print(
        f"BLEU-1={fmt_num(summary.get('open_BLEU_1'))} "
        f"BLEU-2={fmt_num(summary.get('open_BLEU_2'))} "
        f"BLEU-3={fmt_num(summary.get('open_BLEU_3'))} "
        f"BLEU-4={fmt_num(summary.get('open_BLEU_4'))} "
        f"METEOR={fmt_num(summary.get('open_METEOR'))} "
        f"ROUGE-L={fmt_num(summary.get('open_ROUGE_L'))}"
    )

    print("\n[All Valid Rows Text Metrics: MCQ + OPEN]")
    print(
        f"BLEU-1={fmt_num(summary.get('all_BLEU_1'))} "
        f"BLEU-2={fmt_num(summary.get('all_BLEU_2'))} "
        f"BLEU-3={fmt_num(summary.get('all_BLEU_3'))} "
        f"BLEU-4={fmt_num(summary.get('all_BLEU_4'))} "
        f"METEOR={fmt_num(summary.get('all_METEOR'))} "
        f"ROUGE-L={fmt_num(summary.get('all_ROUGE_L'))}"
    )
    print(f"ExactMatch={fmt_pct(summary.get('normalized_exact_match'))} TokenF1={fmt_pct(summary.get('token_f1'))}")
    print(
        f"TokenF1(all)  P={fmt_pct(summary.get('token_precision'))} R={fmt_pct(summary.get('token_recall'))} | "
        f"TokenF1(open)={fmt_pct(summary.get('open_token_f1'))} "
        f"P={fmt_pct(summary.get('open_token_precision'))} R={fmt_pct(summary.get('open_token_recall'))}"
    )

    print("\n[MCQ Detail]")
    print(
        f"SeqMatcher={fmt_pct(summary.get('mcq_seq_accuracy'))} "
        f"Letter={fmt_pct(summary.get('mcq_letter_accuracy'))}"
    )

    print("\n[Inference Time]")
    if summary.get("total_inference_time_sec") is None:
        print("No inference time found in outputs.")
    else:
        print(
            f"Total={summary['total_inference_time_sec']:.2f}s | "
            f"Avg/sample={summary['avg_inference_time_sec']:.2f}s"
        )
    if summary.get("external_wall_time_sec") is not None:
        print(
            f"External wall time={summary['external_wall_time_sec']:.2f}s | "
            f"Avg={summary.get('external_avg_wall_time_sec_per_prediction', 0):.2f}s"
        )
    if summary.get("total_output_tokens") is not None:
        print(
            "\n[Token] "
            f"Total={summary['total_output_tokens']} | "
            f"Avg/question={summary.get('avg_output_tokens_per_question', 0):.2f}"
        )
        print(
            f"Answer={summary.get('total_answer_tokens', 0)} "
            f"(Avg={summary.get('avg_answer_tokens_per_question', 0):.2f}) | "
            f"Explanation={summary.get('total_explanation_tokens', 0)} "
            f"(Avg={summary.get('avg_explanation_tokens_per_question', 0):.2f})"
        )

    if summary.get("num_ttft") or summary.get("num_flops"):
        _g = lambda v: f"{v/1e9:.2f}" if v is not None else "N/A"
        print("\n[Efficiency: KV-cache carry vs re-prefill]")
        print(
            f"TTFT avg={fmt_num(summary.get('avg_ttft_sec'), 4)}s (n={summary.get('num_ttft', 0)}) | "
            f"FLOPs/item avg={_g(summary.get('avg_flops_total'))}G "
            f"(prefill={_g(summary.get('avg_flops_prefill'))}G "
            f"decode={_g(summary.get('avg_flops_decode'))}G)"
        )
        _csr = summary.get("carry_saving_ratio")
        _tsr = summary.get("total_saving_ratio")
        if _csr is not None or summary.get("avg_flops_carry_saving") is not None:
            print(
                f"KV-carry saving={_g(summary.get('avg_flops_carry_saving'))}G "
                f"({_csr*100:.1f}% of append prefill)" if _csr is not None
                else f"KV-carry saving={_g(summary.get('avg_flops_carry_saving'))}G"
            )
            if _tsr is not None:
                print(
                    f"total-cache saving={_g(summary.get('avg_flops_total_saving'))}G "
                    f"({_tsr*100:.1f}% incl. decode cache)"
                )

    if summary.get("coco_metric_warning"):
        print(f"\n[Warning] {summary['coco_metric_warning']}")

    print("\n[Saved]")
    print(f"Report TXT:  {report_path}")
    print(f"Details CSV: {details_path}")
    print(f"Details JSON: {details_json_path}")


def write_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "source_file",
        "question_id",
        "long_id",
        "metadata",
        "answer_type",
        "question",
        "choices_json",
        "ground_truth",
        "prediction",
        "ground_truth_eval",
        "prediction_eval",
        "normalized_exact_match",
        "token_f1",
        "token_precision",
        "token_recall",
        "mcq_seq_correct",
        "mcq_letter_correct",
        "open_substring_correct",
        "inference_time_sec",
        "num_output_tokens",
        "num_answer_tokens",
        "num_explanation_tokens",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    files = collect_files(args.input, args.glob, args.discover_root)
    if not files:
        raise SystemExit("No .json/.jsonl output files found. Pass --input or --glob.")

    references = load_question_gt(args.questions)
    rows = []
    for path in files:
        rows.extend(load_rows_from_file(path))
    rows = fill_missing_gt(rows, references)

    summary, details = evaluate(rows, metric_scope=args.metric_scope)
    summary["num_files"] = len(files)
    summary["input_files"] = files

    external_seconds = args.inference_seconds
    external_count = None
    if args.inference_time_json:
        external_seconds, external_count = load_external_inference_time(args.inference_time_json)
    apply_external_inference_time(summary, external_seconds, external_count)

    original_metrics = load_original_metrics_csv(args.original_metrics_csv)
    if original_metrics is not None:
        summary["original_evaluator"] = original_metrics

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{args.name}_summary.json")
    details_path = os.path.join(args.output_dir, f"{args.name}_details.csv")
    details_json_path = os.path.join(args.output_dir, f"{args.name}_details.json")

    report_path = os.path.join(args.output_dir, f"{args.name}_report.txt")

    write_json(summary_path, summary)
    write_csv(details_path, details)
    write_json(details_json_path, details)

    report_text = build_readable_report(summary)
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    print_readable_summary(summary, report_path, details_path, details_json_path)
    print(f"Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
