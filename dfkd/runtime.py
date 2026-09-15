"""Portable student runtime: worker augmentation, CUDA layouts and mixed precision."""

import inspect
import os
import random
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader


def seed_worker(worker_id):
    random.seed(torch.initial_seed() % (2**32))


class StudentRuntime:
    def __init__(self, device, config):
        self.device = device
        self.config = config
        self.cuda = device.type == "cuda"
        self.channels_last = self.cuda and config.get("channels_last", True)
        precision = config.get("precision", "auto")
        self.dtype = None
        if self.cuda and precision != "fp32":
            self.dtype = (torch.bfloat16 if precision == "bf16" or
                          (precision == "auto" and torch.cuda.is_bf16_supported()) else torch.float16)
            if self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise ValueError("This GPU does not support bf16; use precision=auto or fp32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        workers = config.get("num_workers", "auto")
        self.workers = max(0, min(16, (os.cpu_count() or 2) - 1)) if workers == "auto" else workers
        if self.cuda:
            torch.backends.cudnn.benchmark = config.get("cudnn_benchmark", True)
            torch.backends.cuda.matmul.allow_tf32 = config.get("tf32", True)
            torch.backends.cudnn.allow_tf32 = config.get("tf32", True)

    def prepare(self, model):
        return model.to(memory_format=torch.channels_last) if self.channels_last else model

    def input(self, images):
        images = images.to(self.device, non_blocking=self.cuda)
        return images.contiguous(memory_format=torch.channels_last) if self.channels_last else images

    def autocast(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.dtype else nullcontext()

    def optimizer_options(self, options):
        options = dict(options)
        if (self.cuda and options["name"] == "sgd" and self.config.get("fused_sgd", True)
                and "fused" in inspect.signature(torch.optim.SGD).parameters):
            options["fused"] = True
        return options

    def loader(self, data, batch_size, drop_last, seed):
        options = dict(batch_size=batch_size, shuffle=True, drop_last=drop_last,
                       num_workers=self.workers, pin_memory=self.cuda,
                       generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)
        if self.workers:
            options.update(persistent_workers=True,
                           prefetch_factor=self.config.get("prefetch_factor") or
                           (2 if self.workers <= 2 else 4 if self.workers <= 8 else 6))
        return DataLoader(data, **options)

    def backward_step(self, loss, optimizer):
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()
