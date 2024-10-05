import copy
import dotenv
import dataclasses
import datetime
import os
import time

import torch
import torch.distributed as dist
import tqdm
import wandb
from torch.nn.parallel import DistributedDataParallel
import transformers as tr

from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import VoxtralDataset
from voxtral.trainer.test import test
import voxtral.trainer.train_utils as utils

dotenv.load_dotenv()

Voxtral = tr.MistralForCausalLM

# TODO:
# lora
# prune layers


@dataclasses.dataclass
class TrainState:
    step: int
    model: Voxtral | DistributedDataParallel
    ema: Voxtral
    optimizer: torch.optim.AdamW  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler  # type: ignore
    train_dataset: VoxtralDataset


def init_train_state(config: VoxtralTrainConfig) -> TrainState:
    device = utils.get_device()

    model = Voxtral.from_pretrained(
        config.mistral_pretrained_path,
        **config.mistral_kwargs,
    )
    model = model.train()

    model = model.to(device, torch.bfloat16)

    # initialize ema
    ema = copy.deepcopy(model)
    for param in ema.parameters():
        param.requires_grad = False
    ema = ema.to("cpu")

    optimizer_params = [
        {
            "params": [p for p in model.parameters() if p.dim() > 1],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [p for p in model.parameters() if p.dim() == 1],
            "weight_decay": 0.0,
        },
    ]

    # initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(  # type: ignore
        optimizer_params,
        lr=config.lr,
        betas=config.lr_betas,
        eps=config.lr_eps,
        fused=True if device.type == "cuda" else False,
    )

    scheduler = tr.get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )

    # Initialize dataset
    train_dataset = VoxtralDataset(config)

    # print trainable parameters
    n_params, gb = utils.trainable_params(model)
    utils.pprint(
        f"trainable parameters: {n_params / 1e6:.2f}M | {gb:.2f}GB", color="bold cyan"
    )

    return TrainState(
        step=0,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataset=train_dataset,
    )


def load_train_state(state: TrainState, config: VoxtralTrainConfig) -> TrainState:
    assert config.ckpt_path is not None
    checkpoint = utils._load_checkpoint(config.ckpt_path)
    state.model.load_state_dict(checkpoint["model"])
    state.ema.load_state_dict(checkpoint["ema"])
    state.optimizer.load_state_dict(checkpoint["optimizer"])
    state.scheduler.load_state_dict(checkpoint["scheduler"])
    state.step = checkpoint["step"]
    state.train_dataset.data_step = checkpoint["data_step"]
    utils.pprint(f"loaded from state: {config.ckpt_path}", color="bold yellow")
    return state


def save_state(state: TrainState, config: VoxtralTrainConfig) -> None:
    model = utils.unwrap_model(state.model)
    checkpoint = {
        "model": model.state_dict(),
        "ema": state.ema.state_dict(),
        "optimizer": state.optimizer.state_dict(),
        "scheduler": state.scheduler.state_dict(),
        "step": state.step,
        "data_step": state.step * config.batch_size * config.world_size,
    }
    utils._save_checkpoint(checkpoint, step=state.step, run_id=config.run_id)


def prepare_batch(batch: dict, device: torch.device) -> torch.Tensor:
    return batch["tokens"].to(device)


def compute_loss(
    voxtral: Voxtral | DistributedDataParallel, x: torch.Tensor
) -> torch.Tensor:
    input_ids = x[:, :-1].contiguous()  # Input sequence, excluding the last token
    target_ids = x[:, 1:].contiguous()  # Target sequence, excluding the first token
    outputs = voxtral(input_ids=input_ids)
    logits = outputs.logits
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)).float(), target_ids.view(-1)
    )


def train_step(
    state: TrainState, batch: dict, stats: list[float], config: VoxtralTrainConfig
) -> tuple[TrainState, list[float], dict]:
    device = utils.get_device()
    x = prepare_batch(batch, device)

    loss = compute_loss(state.model, x=x)

    state.optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        state.model.parameters(), config.grad_norm
    )
    state.optimizer.step()
    state.scheduler.step()

    # accumulate stats
    stats[-1] = stats[-1] + 1
    stats[0] = stats[0] + loss.item()
    stats[1] = stats[1] + grad_norm.item()

    ema_info = utils.update_ema_karras_(
        state.ema,
        state.model,
        step=state.step,
        gamma=config.ema_gamma,
        ema_every=config.ema_every,
        log_key="blocks.1.mlp.fc3.weight",
    )

    return state, stats, ema_info


def calculate_throughput(duration: float, batch: dict, hz: int) -> float:
    total_tokens = batch["tokens"].size(1) * batch["tokens"].size(0)
    seconds_of_audio = total_tokens / hz
    return seconds_of_audio / duration


def init_wandb(
    model: Voxtral | DistributedDataParallel, config: VoxtralTrainConfig
) -> None:
    wandb.init(
        config=config.model_dump()
        | {"model_config": utils.unwrap_model(model).config.to_dict()},
        id=config.run_id,
        resume="allow",
        dir=os.path.join(os.getcwd(), "logs"),
        project=config.wandb_project_name,
        name=config.name,
    )
    if config.watch_every is not None:
        wandb.watch(model, log_freq=config.watch_every)


def cleanup():
    utils.rank_0_only(wandb.finish)()
    utils.distributed_only(dist.destroy_process_group)()  # type: ignore


def log_metrics(state: TrainState, stats: list[float]) -> list[float]:
    stats = torch.tensor(stats, device=utils.get_device())  # type: ignore
    utils.distributed_only(dist.all_reduce)(stats, op=dist.ReduceOp.SUM)  # type: ignore

    utils.rank_0_only(wandb.log)(
        {
            "train_loss": (stats[0] / stats[-1]).item(),  # type: ignore
            "current_lr": state.scheduler.get_last_lr()[0],
            "grad_norm": (stats[1] / stats[-1]).item(),  # type: ignore
        },
        step=state.step,
    )
    return torch.zeros_like(stats).tolist()  # type: ignore


def create_loader(
    dataset: VoxtralDataset, config: VoxtralTrainConfig, **kwargs
) -> torch.utils.data.DataLoader:  # type: ignore
    loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
    }
    loader_args.update(kwargs)
    return torch.utils.data.DataLoader(dataset, **loader_args)  # type: ignore


def train(config: VoxtralTrainConfig) -> None:
    utils.pprint(config.model_dump(), json=True)

    utils.distributed_only(dist.init_process_group)(
        "nccl",
        rank=config.rank,
        world_size=config.world_size,
        timeout=datetime.timedelta(seconds=3600),
    )  # type: ignore
    utils.distributed_only(dist.barrier)()  # type: ignore

    utils.set_seed(config.seed)
    state = init_train_state(config)

    if config.ckpt_path:
        state = load_train_state(state, config=config)

    # gradient checkpointing
    if config.gradient_checkpointing:
        state.model.gradient_checkpointing = True  # type: ignore

    # ddp
    if config.world_size > 1:
        state.model = DistributedDataParallel(state.model, device_ids=[config.rank])

    # backend flags
    utils.backend_flags()

    # compile
    if config.compile:
        torch._dynamo.config.optimize_ddp = False  # type: ignore
        state.model = torch.compile(state.model, mode="reduce-overhead")  # type: ignore

    # wandb
    utils.rank_0_only(init_wandb)(state.model, config=config)

    # train loader
    train_loader = iter(create_loader(state.train_dataset, config=config))

    stats = [0.0, 0.0, 0.0]
    train_bar = tqdm.trange(
        config.max_steps - state.step,
        initial=state.step,
        total=config.max_steps,
        colour="blue",
        disable=config.rank > 0,
    )

    for _ in train_bar:
        start_time = time.time()
        try:
            batch = next(train_loader)
        except StopIteration:
            continue

        state.step += 1
        state, stats, ema_info = train_step(
            state, batch=batch, stats=stats, config=config
        )

        end_time = time.time()
        throughput = calculate_throughput(
            end_time - start_time, batch=batch, hz=config.codec_hz
        )

        # log
        train_bar.set_description(
            f"loss: {stats[0]/stats[-1]:.2f} thr: {throughput:.0f}"
        )
        utils.rank_0_only(wandb.log)(
            ema_info, step=state.step
        ) if ema_info is not None else None
        if state.step % 50 == 0:
            stats = log_metrics(state, stats)

        if config.save_every and state.step % config.save_every == 0:
            utils.rank_0_only(save_state)(state, config=config)

        if config.push_every and state.step % config.push_every == 0:
            utils.rank_0_only(state.ema.push_to_hub)(
                f"Audiogen/{config.name}",
                commit_message=f"step {state.step}, run_id {config.run_id}",
                private=True,
            )

        if config.test_every and state.step % config.test_every == 0:
            test(state.ema, step=state.step, config=config)

        if state.step >= config.max_steps:
            utils.pprint("\nmax steps reached, exiting...", color="bold red")
            break

        utils.distributed_only(dist.barrier)()  # type: ignore

    cleanup()
