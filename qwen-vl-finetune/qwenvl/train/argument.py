import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    model_type: str = field(default="qwen3vl")
    images_in_zip: bool = field(default=False)
    datamix_config_yml: str = field(default="")
    datamix_path_regex: str = field(default="") # if set, will override the field in datamix_config_yml (all datasets)
    annotation_path: str = field(default="")
    data_path: str = field(default="")
    sampling_rate: float = field(default=1.0)
    no_shuffle: bool = field(default=False) # if set, do not shuffle the dataset, for controlled experiments
    train_set_size: Optional[int] = field(default=None) # if set, enforce exact number of training samples to load from dataset; for controlling correct resuming in chunked training
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
    resume_from_other_train_output_dir: Optional[str] = field(default=None) # default resume from output_dir, if not None, resume from this dir
    is_chunked_training: bool = field(default=False) # whether to use chunked training
    is_within_chunk_resume: bool = field(default=False) # whether to resume inside a chunk. False if a fresh resuming from previous chunk. Set this to precisely control data skipping. Resuming inside a chunk requires data skipping while fresh resuming from previous chunk does not.
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
    lora_target_modules: str = field(default="q_proj,k_proj,v_proj") # Other popular choices: "q_proj,k_proj,v_proj,gate_proj,up_proj,down_proj,o_proj"

    ## for debug
    print_model: Optional[bool] = False
    print_trainable_params: Optional[bool] = False # for lora

    def __post_init__(self):
        # transformers.TrainingArguments has its own __post_init__, so we need to call it
        super().__post_init__()
        if self.min_lr is not None:
            self.lr_scheduler_kwargs = self.lr_scheduler_kwargs or {}
            self.lr_scheduler_kwargs["min_lr"] = self.min_lr
        if self.lora_enable and self.lora_target_modules:
            self.lora_target_modules = [x.strip() for x in self.lora_target_modules.split(",")]

        if self.resume_from_other_train_output_dir and self.is_chunked_training and not self.is_within_chunk_resume:
            self.ignore_data_skip = True # no data skipping for fresh resuming from previous chunk

