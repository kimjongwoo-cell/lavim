"""Typed canonical experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal


DatasetName = Literal["wsi-vqa", "expertvqa", "slidebench", "tcga", "gtex", "panda"]
MethodName = Literal[
    "single",
    "single-no-thinking",
    "vlmas",
    "latent-base",
    "pruning-v3",
    "reallocation-v2",
    "both",
]
ModelName = Literal["qwen-2b", "qwen-4b", "qwen-8b"]
EngineRole = Literal["single", "text-mas", "latent-mas"]


@dataclass(frozen=True, slots=True)
class MethodDefinition:
    """Map one public method tag to its execution engine and latent variant."""

    engine_role: EngineRole
    latent_variant: Literal["base", "pruning_v3", "reallocation_v2", "both"] | None = None


METHOD_DEFINITIONS: Final[dict[MethodName, MethodDefinition]] = {
    "single": MethodDefinition(engine_role="single"),
    "single-no-thinking": MethodDefinition(engine_role="single"),
    "vlmas": MethodDefinition(engine_role="text-mas"),
    "latent-base": MethodDefinition(engine_role="latent-mas", latent_variant="base"),
    "pruning-v3": MethodDefinition(engine_role="latent-mas", latent_variant="pruning_v3"),
    "reallocation-v2": MethodDefinition(engine_role="latent-mas", latent_variant="reallocation_v2"),
    "both": MethodDefinition(engine_role="latent-mas", latent_variant="both"),
}

CODE_ROOT: Final = Path(__file__).resolve().parents[2]
PROJECT_ROOT: Final = CODE_ROOT.parent
ASSET_ROOT: Final = PROJECT_ROOT.parent
MODEL_ROOT: Final = ASSET_ROOT / "models"
WSIVQA_ROOT: Final = ASSET_ROOT / "datasets" / "WSI-VQA"
MULTIPATH_ROOT: Final = ASSET_ROOT / "datasets" / "MultiPathQA" / "ready_wsivqa"
RUN_ROOT: Final = PROJECT_ROOT / "results" / "runs" / "canonical"


@dataclass(frozen=True, slots=True)
class HyperParameters:
    """Every tunable final-run setting, controlled only by ``run.py``."""

    max_model_len: int = 8192
    patch_budget: int = 8
    navigator_control_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.95
    seed: int = 42
    max_new_tokens: int = 512
    cpu_threads: int = 4
    single_case_workers: int = 1
    io_pipeline: bool = True
    navigator_kv: bool = True
    canonical_open_options: bool = True
    deterministic: bool = True
    greedy_decoding: bool = True
    single_do_sample: bool = True
    single_reasoner_style: bool = True
    single_direct_final_only: bool = False
    answerer_thinking: bool = False
    answerer_rationale: bool = True
    answerer_protocol: Literal["structured_json", "tag"] = "structured_json"
    transport_mode: Literal["cumulative", "cumulative_nothink"] = "cumulative"
    realign_method: Literal["identity_norm", "wa", "softmax"] = "wa"
    fast_json: bool = True
    constrained_json: bool = False
    save_navigation_pngs: bool = False
    terminal_no_repeat_ngram_size: int = 0
    terminal_repetition_penalty: float = 1.0
    terminal_anti_repeat: Literal["none", "boxed_ngram3", "boxed_ngram3_penalty", "tag_ngram3", "json_ngram3"] = "none"
    case_retries: int = 0

    @property
    def effective_temperature(self) -> float:
        """Return the decoding temperature after the shared greedy policy."""
        return 0.0 if self.greedy_decoding else self.temperature

    @property
    def effective_top_p(self) -> float:
        """Return the sampling nucleus parameter after the shared greedy policy."""
        return 1.0 if self.greedy_decoding else self.top_p


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One reproducible final experiment selected from stable tags."""

    dataset: DatasetName
    model: ModelName
    method: MethodName
    steps: int
    gpus: tuple[int, ...]
    output_root: Path
    resume: bool
    dry_run: bool
    hyperparameters: HyperParameters

    @property
    def method_definition(self) -> MethodDefinition:
        """Return the engine definition selected by the public method tag."""
        return METHOD_DEFINITIONS[self.method]

    @property
    def engine_role(self) -> EngineRole:
        """Return the one engine responsible for this experiment."""
        return self.method_definition.engine_role

    @property
    def latent_variant(self) -> Literal["base", "pruning_v3", "reallocation_v2", "both"]:
        """Return the latent variant for methods routed to the latent engine."""
        variant = self.method_definition.latent_variant
        if variant is not None:
            return variant
        raise ValueError(f"method does not use a latent variant: {self.method}")

    @property
    def model_path(self) -> Path:
        names: Final = {
            "qwen-2b": "Qwen3-VL-2B-Thinking",
            "qwen-4b": "Qwen3-VL-4B-Thinking",
            "qwen-8b": "Qwen3-VL-8B-Thinking",
        }
        return MODEL_ROOT / names[self.model]

    @property
    def dataset_path(self) -> Path:
        if self.dataset == "wsi-vqa":
            return WSIVQA_ROOT / "WsiVQA_test.json"
        filenames: Final = {
            "expertvqa": "tcga_expert_vqa.json",
            "slidebench": "tcga_slidebench.json",
            "tcga": "tcga.json",
            "gtex": "gtex.json",
            "panda": "panda.json",
        }
        return MULTIPATH_ROOT / "full_no_panda" / filenames[self.dataset]

    @property
    def slide_root(self) -> Path:
        if self.dataset == "wsi-vqa":
            return WSIVQA_ROOT / "DATA_SVS"
        return MULTIPATH_ROOT / "full_no_panda" / "slides"

    @property
    def contract_path(self) -> Path:
        if self.dataset == "wsi-vqa":
            return CODE_ROOT / "config" / "wsivqa_fair_v2.json"
        return MULTIPATH_ROOT / "wsivqa_fair_v2_multipathqa.json"

    @property
    def thumbnail_seed_dir(self) -> Path | None:
        if self.dataset == "wsi-vqa":
            return None
        return MULTIPATH_ROOT / "full_no_panda" / "thumbnails_sanitized"


def canonical_output_root(
    dataset: DatasetName,
    model: ModelName,
    method: MethodName,
    steps: int,
) -> Path:
    """Return the only default output layout for final experiments."""
    suffix = method if method in {"single", "single-no-thinking", "vlmas"} else f"{method}-step{steps}"
    return RUN_ROOT / dataset / model / suffix
