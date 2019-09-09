from argparse import ArgumentParser
import os
from pathlib import Path
import shutil
import glob
import logging
import json
import random
import numpy as np
from tempfile import TemporaryDirectory
from collections import namedtuple
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pytorch_transformers.modeling_utils import WEIGHTS_NAME

def init(args):
    # init logger
    log_format = '%(asctime)-10s: %(message)s'
    if args.log_file is not None and args.log_file != "":
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w', format=log_format)
        logging.warning(f'This will get logged to file: {args.log_file}')
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    # create output dir
    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    assert 'bert' in args.output_dir.name, \
        '''Output dir name has to contain `bert` or `roberta` for AutoModel.from_pretrained to correctly infer the model type'''

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_args_parser_with_general_args():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)

    parser.add_argument("--bert_model", type=str, required=True, help="Bert pre-trained model. Either a path to the model dir or "
                             "selected from list: bert-base-uncased, bert-large-uncased, bert-base-cased, "
                             "bert-base-multilingual, bert-base-chinese, roberta-base, roberta-large")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--warmup_steps", 
                        default=0, 
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon", 
                        default=1e-8, 
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--log-file', default=None, type=str)
    return parser


def save_checkpoint(model, epoch, output_dir):
    weights_name, ext = os.path.splitext(WEIGHTS_NAME)
    save_comment=f'{epoch:04d}'
    weights_name += f'-{save_comment}{ext}'
    output_model_file = os.path.join(output_dir, weights_name)
    logging.info(f"Saving fine-tuned model to: {output_model_file}")
    torch.save(model.state_dict(), output_model_file)


def prepare_last_checkpoint(pretrained_model_name_or_path):
    if not os.path.isdir(pretrained_model_name_or_path):
        return 0  # It is probabaly a model name, not an input directory

    weights_name, ext = os.path.splitext(WEIGHTS_NAME)
    archive_files = sorted(glob.glob(f'{pretrained_model_name_or_path}/{weights_name}*{ext}'))
    if len(archive_files) > 1 and archive_files[-1].endswith(WEIGHTS_NAME):
        archive_file = archive_files[-2]  # if the last file is `pytorch_model.bin`, ignore it and use the one before
    else:
        archive_file = archive_files[-1]
    logging.info(f'Found {len(archive_files)} model files. Use the most recent, {archive_file}')

    # extract epoch number  (some/dir/pytorch_model-epochNumber.bin or some/dir/pytorch_model.bin
    filename = archive_file.split('/')[-1]
    assert filename.startswith(weights_name)
    filename_without_ext = filename.split('.')[0]
    splits = filename_without_ext.split('-')
    if len(splits) == 1:
        start_epoch = 0  # filename is `pytorch_model.bin`, do nothing
    elif len(splits) == 2:
        # filename is `pytorch_model-epochNumber.bin`
        assert splits[0] == weights_name
        # read epoch number to continue training from the last point
        start_epoch = int(splits[1]) + 1
        # copy `pytorch_model-epochNumber.bin` to `pytorch_model.bin`
        # because that's what the `from_pretrained` is loading from
        dest_filename = archive_file.replace(filename, WEIGHTS_NAME)
        logging.info(f'For loading, copy {archive_file} to {dest_filename}')
        shutil.copy(archive_file, dest_filename)
    else:
        assert False  # wrong name
    return start_epoch


def get_dataset_stats(args, n_tpu):
    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    total_train_examples = 0
    for i in range(args.start_epoch, args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(total_train_examples / args.train_batch_size / n_tpu)
    return num_data_epochs, num_train_optimization_steps


InputFeatures = namedtuple("InputFeatures", "input_ids input_mask segment_ids lm_label_ids is_next")


def convert_example_to_features(example, tokenizer, max_seq_length):
    tokens = example["tokens"]
    segment_ids = example["segment_ids"]
    is_random_next = example["is_random_next"]
    masked_lm_positions = example["masked_lm_positions"]
    masked_lm_labels = example["masked_lm_labels"]

    assert len(tokens) == len(segment_ids) <= max_seq_length  # The preprocessed data should be already truncated
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

    input_array = np.zeros(max_seq_length, dtype=np.int)
    input_array[:len(input_ids)] = input_ids

    mask_array = np.zeros(max_seq_length, dtype=np.bool)
    mask_array[:len(input_ids)] = 1

    segment_array = np.zeros(max_seq_length, dtype=np.bool)
    segment_array[:len(segment_ids)] = segment_ids

    lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=-1)
    lm_label_array[masked_lm_positions] = masked_label_ids

    features = InputFeatures(input_ids=input_array,
                             input_mask=mask_array,
                             segment_ids=segment_array,
                             lm_label_ids=lm_label_array,
                             is_next=is_random_next)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False):
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            segment_ids = np.memmap(filename=self.working_dir/'segment_ids.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = -1
            is_nexts = np.memmap(filename=self.working_dir/'is_nexts.memmap',
                                 shape=(num_samples,), mode='w+', dtype=np.bool)
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            segment_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=-1)
            is_nexts = np.zeros(shape=(num_samples,), dtype=np.bool)
        logging.info(f"Loading training examples for epoch {epoch} from {data_file}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                segment_ids[i] = features.segment_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids
                is_nexts[i] = features.is_next
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logging.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids
        self.is_nexts = is_nexts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.segment_ids[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.is_nexts[item].astype(np.int64)))
