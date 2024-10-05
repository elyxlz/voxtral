import typing
import random
import numpy as np
import collections
import re
import glob
import huggingface_hub as hf_hub
import rich.console
import os
import traceback
import sys
import torch


console = rich.console.Console()


def general_exception_handler(func: typing.Callable) -> typing.Callable:
    """training script must never die"""

    def inner_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"\n{'=' * 40}")
            pprint(f"error in function '{func.__name__}':", color="bold red")
            print(f"{'=' * 40}")
            print(f"exception type: {type(e).__name__}")
            print(f"exception message: {str(e)}")
            print("\ntraceback:")
            traceback.print_exc(file=sys.stdout)
            print(f"{'=' * 40}\n")

    return inner_function


def rank_0_only(func: typing.Callable) -> typing.Callable:
    def inner_function(*args, **kwargs):
        if int(os.getenv("RANK", 0)) == 0:
            return func(*args, **kwargs)

    return inner_function


def local_rank_0_only(func: typing.Callable) -> typing.Callable:
    def inner_function(*args, **kwargs):
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            return func(*args, **kwargs)

    return inner_function


def distributed_only(func: typing.Callable) -> typing.Callable:
    def inner_function(*args, **kwargs):
        if int(os.getenv("WORLD_SIZE", 1)) > 1:
            return func(*args, **kwargs)

    return inner_function


def reduce_float(x: float) -> float:
    x = torch.tensor(x, device=get_device())  # type: ignore
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)  # type: ignore
    return x.item() / int(os.getenv("WORLD_SIZE", 1))  # type: ignore


def get_device() -> torch.device:
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


@local_rank_0_only
def pprint(
    item: str | dict, *args, color: str | None = None, json: bool = False, **kwargs
) -> None:
    if json:
        console.print_json(data=item, *args, **kwargs)
    else:
        item = f"[{color}]{item}[/{color}]" if color else item
        console.print(item, *args, **kwargs)


def trainable_params(model: torch.nn.Module) -> tuple[int, float]:
    first_param = next(model.parameters())
    dtype_size = first_param.element_size()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_in_bytes = trainable_params * dtype_size
    size_in_gb = size_in_bytes / (1024**3)
    return trainable_params, size_in_gb


def _load_checkpoint(path: str, cpu: bool = False) -> dict:
    if "huggingface.co" in path:
        split_path = path.split("/")
        repo_id = "".join(split_path[1:3])
        filename = split_path[-1]
        path = hf_hub.hf_hub_download(repo_id, filename=filename)
    device = "cpu" if cpu else get_device()
    return torch.load(path, map_location=device, weights_only=False)


def unwrap_model(
    model: torch.nn.Module
    | torch.nn.parallel.DistributedDataParallel
    | torch._dynamo.eval_frame.OptimizedModule,
) -> torch.nn.Module:
    if isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
        model = model._orig_mod
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    return model


@general_exception_handler
def _save_checkpoint(
    checkpoint: dict[str, typing.Any], step: int, run_id: str, push: bool = False
) -> None:
    """serializes a dict to the logs directory with torch, only keeps 5 most recent checkpoints in dir"""
    log_dir = os.path.join(os.path.join(os.getcwd(), "logs"), run_id)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_path = os.path.join(log_dir, f"checkpoint_{step}.pt")
    torch.save(checkpoint, checkpoint_path)

    checkpoint_files = glob.glob(os.path.join(log_dir, "checkpoint_*.pt"))

    def get_step_number(filename):
        match = re.search(r"checkpoint_(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    checkpoint_files.sort(key=get_step_number, reverse=True)
    for old_checkpoint in checkpoint_files[5:]:
        os.remove(old_checkpoint)

    if push:
        api = hf_hub.HfApi()

        api.create_repo(
            repo_id=f"Audiogen/{run_id}",
            private=True,
            token=os.getenv("HUGGINGFACE_TOKEN"),
            exist_ok=True,
        )

        future = api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"checkpoint_{step}.pt",
            repo_id=f"Audiogen/{run_id}",
            commit_message=f"{step}",
            run_as_future=True,
        )
        future.result()
        future.done()

    pprint(f"saved state in path {checkpoint_path}", color="bold yellow")


@torch.no_grad()
def update_ema_karras_(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    step: int,
    gamma: float,
    ema_every: int = 1,
    log_key: str | None = None,
) -> dict:
    step = max(step, 1)
    if step % ema_every != 0:
        return {}

    ema_device = next(ema_model.parameters()).device
    ema_params = collections.OrderedDict(ema_model.named_parameters())
    model_params = collections.OrderedDict(model.named_parameters())

    BETA = (1 - 1 / step) ** (1 + gamma)
    BETA = 1 - (1 - BETA) * ema_every
    BETA = max(0, BETA)

    info = {}
    info["ema_decay"] = BETA

    for k, v in model_params.items():
        # for torch compile or ddp
        if k.startswith("_orig_mod."):
            k = k.replace("_orig_mod.", "")
        if k.startswith("module."):
            k = k.replace("module.", "")

        ema_params[k].mul_(BETA).add_(v.to(ema_device), alpha=1 - BETA)

        if log_key is not None and k == log_key:
            info["real_param"] = v.flatten()[0].item()
            info["ema_param"] = ema_params[k].flatten()[0].item()

    return info


def backend_flags() -> None:
    from torch.backends import cuda, cudnn

    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.allow_tf32 = True
    cuda.matmul.allow_tf32 = True
    cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("medium")
    torch._dynamo.config.cache_size_limit = max(  # type: ignore
        128,
        torch._dynamo.config.cache_size_limit,  # type: ignore
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
