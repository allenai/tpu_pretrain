
import time
import logging
import random
import numpy as np
from tqdm import tqdm

import utils  # Most of the code in adopted from https://github.com/huggingface/pytorch-transformers/blob/master/examples/lm_finetuning/finetune_on_pregenerated.py
from pytorch_transformers.modeling_auto import AutoModelWithLMHead
from pytorch_transformers.tokenization_auto import AutoTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

from torch.utils.data import DataLoader, RandomSampler
import torch
import torch_xla
import torch_xla_py.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.data_parallel as dp
import torch_xla.distributed.parallel_loader as pl


from pytorch_transformers.modeling_roberta import RobertaModel
def RobertaModel_forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
    return super(RobertaModel, self).forward(input_ids, token_type_ids, attention_mask, position_ids, head_mask)
RobertaModel.forward = RobertaModel_forward  # RobertaModel.forward has a `.item()` which doesn't work nicely with TPUs

parser = utils.get_args_parser_with_general_args()
parser.add_argument('--one_tpu', action='store_true', help="Run on one tpu core for degugging. Makes it easy to use break points")
parser.add_argument('--tpu_report', action='store_true', help="Print xla metric report")
parser.add_argument('--num_cores', default=8, type=int, help="Number of tpu cores to use")
args = parser.parse_args()
args.start_epoch = 0


def main():
    utils.init(args)  # set seeds, init logger, prepare output directory

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    tokenizer.save_pretrained(args.output_dir)

    args.start_epoch = 0 # utils.prepare_last_checkpoint(args.bert_model)  TODO: fix
    model = AutoModelWithLMHead.from_pretrained(args.bert_model)  # Only Masked Language Modeling
    logging.info(f"Saving initial checkpoint to: {args.output_dir}")

    device = xm.xla_device()
    model = model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # one optimizer and scheduler per TPU core. Both objects are saved in `context` to be reused the next epoch
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=tuple(args.betas))
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=10)

    n_tpu = 1
    num_data_epochs, num_train_optimization_steps= utils.get_dataset_stats(args, n_tpu)

    def tpu_training_loop(loader):
        """ Called by torch_xla_py.data_parallel. This function is executed on each core of the TPU once per epoch"""

        tr_loss = None
        pbar = None
        if str(pbar_device) == str(device):  # All threads are in sync. Use progress bar only on one of them
            pbar = tqdm(total=int(pbar_steps), desc=f"device {device}", dynamic_ncols=True)

        tracker = xm.RateTracker()

        model.train()
        step = 0
        for input_ids, input_mask, segment_ids, lm_label_ids, _ in loader:
            outputs = model(input_ids, segment_ids, input_mask, lm_label_ids)
            loss = outputs[0]
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tracker.add(args.train_batch_size)
            
            tr_loss = loss * args.gradient_accumulation_steps if step == 0 else  tr_loss + loss * args.gradient_accumulation_steps
            if pbar is not None:
                pbar.update(1)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
            step += 1
            if step == 4: 
                logging.info(torch_xla._XLAC._xla_metrics_report())

            
        return tr_loss.item()/step  # `.item()` requires a trip from TPU to CPU, which is very slow. Use it only once per epoch=

    for epoch in range(args.start_epoch, args.epochs):
        # Load one training file into memory
        epoch_dataset = utils.PregeneratedDataset(epoch=epoch, training_path=args.pregenerated_data, tokenizer=tokenizer,
                                            num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        train_sampler = RandomSampler(epoch_dataset)
        train_dataloader = torch.utils.data.DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        para_loader = pl.ParallelLoader(train_dataloader, [device])

        pbar_device = device
        pbar_steps = int(train_sampler.num_samples / args.train_batch_size / n_tpu)

        # logging.info(f'start training, epoch {epoch} on {len(devices)} cores for {pbar_steps} steps')
        start = time.time()
        losses = tpu_training_loop(para_loader.per_device_loader(device))

        logging.info(f'Epoch {epoch} took {round(time.time() - start, 2)} seconds. Average loss: {sum(losses)/len(losses)}')
        utils.save_checkpoint(model._models[0], epoch, args.output_dir)

    if args.tpu_report:
        logging.info(torch_xla._XLAC._xla_metrics_report())


def _mp_fn(index, flags):
  global args
  args = flags
  torch.set_default_tensor_type('torch.FloatTensor')
  main()


if __name__ == '__main__':
  # xmp.spawn(_mp_fn, args=(args,), nprocs=args.num_cores)
  main()