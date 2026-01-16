import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    resume_from_other_train_output_dir: Optional[str] = field(default=None) # default resume from output_dir, if not None, resume from this dir
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    model_type: str = field(default="qwen3vl")
    images_in_zip: bool = field(default=False)
    annotation_path: str = field(default="")
    data_path: str = field(default="")
    sampling_rate: float = field(default=1.0)
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    chunk_stop_steps: Optional[int] = field(default=None) # used for chunk training, stop steps for each chunk
    callback_report_to: str|List[str] = field(default="csv") # default to csv
    # optim: str = field(default="adamw_torch")
    min_lr: float = field(
        default=1e-5, metadata={"help": "The minimum learning rate to use."}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

    def __post_init__(self):
        # transformers.TrainingArguments has its own __post_init__, so we need to call it
        super().__post_init__()
        if self.min_lr is not None:
            self.lr_scheduler_kwargs = self.lr_scheduler_kwargs or {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_lr
