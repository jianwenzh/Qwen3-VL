# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
from typing import List
import mlflow
import torch
import transformers
import sys
from pathlib import Path
import signal
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from trainer import replace_qwen2_vl_attention_class
from transformers.trainer_utils import get_last_checkpoint

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    TrainerCallback
)
from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor, Trainer
from ..common.state import rank0_print, set_local_rank

class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_step: int, should_save: bool = True):
        self.stop_step = stop_step
        self.should_save = should_save

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
            control.should_save = self.should_save
            if state.is_local_process_zero:  # only log on main process
                rank0_print(f"[StopAtStepCallback]: Set stopping training at step {state.global_step}")

        return control

_CSV_LOG_FILE_NAME = "metrics_log.txt"
class CsvLogCallback(TrainerCallback):
    """
    Vanilla [`TrainerCallback`] that sends the logs to CSV file
    """
    def __init__(self, output_dir: str):
        self.log_file_path = os.path.join(output_dir, _CSV_LOG_FILE_NAME)
        with open(self.log_file_path, "a", encoding="utf-8") as fo:
            fo.write("step,key,value\n")

    def log_metric(self, key, value, step):
        with open(self.log_file_path, "a", encoding="utf-8") as fo:
            fo.write(f"{step},{key},{value}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.log_metric(k, v, step=state.global_step)

class AzureMLV2LogCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [AzureML via v2 SDK](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2&tabs=interactive)
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v, step=state.global_step)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def print_trainable_param_names(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            rank0_print(name, param.shape)

def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def set_callbacks(training_args: TrainingArguments) -> List[TrainerCallback]:
    callbacks = []
    if training_args.callback_report_to is not None:
        for rpt in training_args.callback_report_to:
            if rpt == 'csv':
                callbacks.append(CsvLogCallback(output_dir=training_args.output_dir))
            elif rpt == 'azure_ml_v2':
                callbacks.append(AzureMLV2LogCallback())
            else:
                raise ValueError(f"Unknown callback_report_to: {rpt}")
    
    if training_args.chunk_stop_steps is not None:
        callbacks.append(StopAtStepCallback(stop_step=training_args.chunk_stop_steps, should_save=True))
    
    if len(callbacks) == 0:
        callbacks = None
    return callbacks


def train(attn_implementation="flash_attention_2"):
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_local_rank(training_args.local_rank)

    rank0_print("Model arguments:", model_args)
    rank0_print("Data arguments:", data_args)
    rank0_print("Training arguments:", training_args)

    os.makedirs(training_args.output_dir, exist_ok=True)

    # if "qwen3" in model_args.model_name_or_path.lower() and "a" in Path(model_args.model_name_or_path.rstrip("/")).name.lower():
    if data_args.model_type == "qwen3vlmoe":
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    # elif "qwen3" in model_args.model_name_or_path.lower():
    elif data_args.model_type == "qwen3vl":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen3vl"
    # elif "qwen2.5" in model_args.model_name_or_path.lower():
    elif data_args.model_type == "qwen2.5vl":
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2.5vl"
    # else:
    elif data_args.model_type == "qwen2vl":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bf16 else None),
        )
        data_args.model_type = "qwen2vl"
    else:
        raise NotImplementedError(f"Model type {data_args.model_type} not supported yet.")

    rank0_print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )

    if training_args.print_model:
        rank0_print(model)
    
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        rank0_print("LoRA enabled")

        for p in model.parameters():
            p.requires_grad = False

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=training_args.lora_target_modules, # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if training_args.print_trainable_params:
            print_trainable_param_names(model)
        
        model.print_trainable_parameters()
    else:
        set_model(model_args, model)

        if torch.distributed.get_rank() == 0:
            model.visual.print_trainable_parameters()
            model.model.print_trainable_parameters()

    data_module = make_supervised_data_module(processor, data_args=data_args)
    callbacks = set_callbacks(training_args)
    rank0_print(f"Using callbacks: {callbacks}")
    trainer = Trainer(
        model=model, processing_class=tokenizer, args=training_args, callbacks=callbacks, **data_module
    )

    
    if model_args.resume_from_other_train_output_dir is not None:
        if not os.path.exists(model_args.resume_from_other_train_output_dir) or not os.path.isdir(model_args.resume_from_other_train_output_dir):
            raise ValueError(f"model_args.resume_from_other_train_output_dir {model_args.resume_from_other_train_output_dir} does not exist or is not a directory")
        
        resume_from_checkpoint = get_last_checkpoint(model_args.resume_from_other_train_output_dir)
        if resume_from_checkpoint is None:
            raise ValueError(f"No valid checkpoint found in specified resume_from_other_train_output_dir directory ({model_args.resume_from_other_train_output_dir})")
        
        logging.info(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    elif training_args.output_dir and list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found from output_dir, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:        
        trainer.train()
    
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


def handler(sig, frame):
    print("Caught SIGINT, exiting...")
    try:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    except Exception:
        pass
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)  # optional
    train(attn_implementation="flash_attention_2")
    