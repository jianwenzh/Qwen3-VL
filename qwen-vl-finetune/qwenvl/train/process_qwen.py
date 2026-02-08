
import copy
from dataclasses import field, dataclass
import glob
import json
import os
import logging
import shutil
from typing import List
import transformers
import sys
from pathlib import Path
import sys
import time
import traceback

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor import LazySupervisedDataset, DataCollatorForSupervisedDataset
from qwenvl.train.argument import (
    DataArguments,
)
from transformers import AutoProcessor

logging.basicConfig(
    format="[%(prefix)s] %(asctime)s - %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

_base_logger = logging.getLogger(__name__)
_main_logger = logging.LoggerAdapter(_base_logger, {'prefix': '[MAIN]'})

@dataclass
class ProcessArgs:
    model_dir: str = field(default="")
    output_dir: str = field(default="")
    annotations_root_dir: str = field(default="")
    images_root_dir: str = field(default="")
    glob_patterns: str = field(default="")
    n_proc: int = field(default=-1)
    # packing: bool = field(default=False) # doesn't support packing now

def process_file(
    annotations_root_dir: str,
    images_root_dir: str,
    relative_path: str,
    output_dir: str,
    data_args: ProcessArgs,
    process_args: ProcessArgs,
):
    logger = logging.LoggerAdapter(_base_logger, {'prefix': f'[{relative_path}]'})
    # first check if output file exists, if yes, copy a bak, load, check corruption, and resume
    # if no output file, create new
    output_file_path = os.path.join(output_dir, relative_path)
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    n_processed_lines = 0
    n_processed_errors = 0
    if os.path.exists(output_file_path) and os.path.isfile(output_file_path):
        logger.info(f"Output file {output_file_path} exists, check for corrupted lines and resume processing.")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        bak_file = output_file_path + f".bak.{timestamp}"
        shutil.copyfile(output_file_path, bak_file)
        with open(bak_file, "r", encoding="utf-8") as fin, open(output_file_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                try:
                    jobj = json.loads(line)
                    if "error" in jobj:
                        n_processed_errors += 1
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted line found: line {i}, stop loading more.")
                    continue
                fout.write(line)
                n_processed_lines += 1
    if n_processed_lines > 0:
        logger.info(f"Resumed processing file {relative_path}, found {n_processed_lines} processed lines, which will be skipped.")
    
    # read from annotations file, skip processed lines
    annotations_file_path = os.path.join(annotations_root_dir, relative_path)
    images_zip_file_path = os.path.join(images_root_dir, relative_path.replace(".jsonl", ".zip"))
    remain_annotations_file = annotations_file_path
    if n_processed_lines > 0:
        remain_annotations_file = "./remain_annotations.tmp.jsonl"
        n_remain_lines = 0
        with open(annotations_file_path, "r", encoding="utf-8") as fin, open(remain_annotations_file, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i < n_processed_lines:
                    continue
                fout.write(line)
                n_remain_lines += 1
        logger.info(f"Created temporary remain annotations file {remain_annotations_file} with {n_remain_lines} lines for processing.")
    
    # load all remaining examples
    logger.info(f"Loading remaining examples from {remain_annotations_file}")
    remain_examples = []
    with open(remain_annotations_file, "r", encoding="utf-8") as fin:
        for line in fin:
            jobj = json.loads(line)
            remain_examples.append(jobj)
    n_remain = len(remain_examples)
    logger.info(f"Total {n_remain} remaining examples to process in {relative_path}")

    # loading processor
    processor = AutoProcessor.from_pretrained(
        process_args.model_dir,
    )
    data_args_copy = copy.deepcopy(data_args)
    data_args_copy.annotation_path = remain_annotations_file
    data_args_copy.data_path = images_zip_file_path
    data_args_copy.sampling_rate = 1.0  # do not sample during processing
    data_args_copy.no_shuffle = True  # do not shuffle during processing
    data_args_copy.images_in_zip = True  # images are in zip during processing
    data_args_copy.data_packing = False  # disable data packing during processing
    data_args_copy.data_flatten = False  # disable data flattening during processing
    
    # create dataset
    dataset = LazySupervisedDataset(
        processor=processor,
        data_args=data_args_copy,
    )
    n_total = len(dataset)
    if n_total != n_remain:
        raise ValueError(f"Loaded dataset size {n_total} does not match remaining annotations size {n_remain}")
    
    logger.info(f"Dataset loaded with {n_total} samples to process")

    collator = DataCollatorForSupervisedDataset(tokenizer=processor.tokenizer)
    with open(output_file_path, "a", encoding="utf-8") as fout:
        n_errors = 0
        for idx in range(n_total):
            error = None
            n_tokens = None
            orig_example = remain_examples[idx]
            try:
                example = dataset[idx]
                batch = collator([example])
            except Exception as e:
                tb = traceback.format_exc()
                error = f"{e}\n{tb}"
                logger.warning(f"Error processing example idx {idx} in file {relative_path}: {error}. Orig example: {orig_example}")
                n_errors += 1
            # get token length
            input_ids = batch["input_ids"]
            n_tokens = input_ids.size(1)
            output_obj = copy.deepcopy(orig_example)
            output_obj["n_tokens"] = n_tokens # write even if null, incase if legacy value existing
            if error is not None:
                output_obj["error"] = error
            fout.write(json.dumps(output_obj) + "\n")
            fout.flush()
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx} / {n_total} examples, n_errors: {n_errors}, total_examples: {n_processed_lines + idx}, total_errors: {n_processed_errors + n_errors}")

    logger.info(f"Finished processing file {relative_path}, n_examples this run: {n_total}, total examples: {n_total + n_processed_lines}, errors in this run: {n_errors}, total errors: {n_errors + n_processed_errors}")


def glob_files(dir: str, glob_patterns: str) -> List[str]:
    files = set()
    for pattern in glob_patterns.split('|'):
        files.update(glob.glob(os.path.join(dir, pattern), recursive=True))

    files = [f for f in files if os.path.isfile(f)]
    return sorted(list(files))


def main():
    parser = transformers.HfArgumentParser((ProcessArgs, DataArguments))
    process_args, data_args = parser.parse_args_into_dataclasses()
    annotation_files = glob_files(dir=process_args.annotations_root_dir, glob_patterns=process_args.glob_patterns)
    if len(annotation_files) == 0:
        raise ValueError("Not found any annotation files")
    _main_logger.info(f"Found {len(annotation_files)} annotation files")
    args = []
    for annotation_file in annotation_files:
        relative_path = os.path.relpath(annotation_file, process_args.annotations_root_dir)
        args.append((process_args.annotations_root_dir, process_args.images_root_dir, relative_path, process_args.output_dir, data_args, process_args))

    n_proc = process_args.n_proc
    if n_proc <= 0:
        n_proc = os.cpu_count()
    n_proc = min(n_proc, len(args))
    _main_logger.info(f"Using {n_proc} processes for processing files")
    if n_proc == 1:
        for arg in args:
            process_file(*arg)
    else:
        from multiprocessing import Pool
        with Pool(n_proc) as p:
            p.starmap(process_file, args)

    
if __name__ == "__main__":
    main()