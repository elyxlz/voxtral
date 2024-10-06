import itertools

import numpy as np
import torch
import torch.distributed
import tqdm
import transformers as tr
import voxtral.trainer.utils as utils
import wandb
from voxtral.tokenizer.model import VoxtralTokenizer
from voxtral.trainer.config import VoxtralTrainConfig
from voxtral.trainer.data import VoxtralDataset

Voxtral = tr.MistralForCausalLM


@utils.general_exception_handler
@torch.no_grad()
def test(ema_model: Voxtral, step: int, config: VoxtralTrainConfig) -> None:
    from .trainer import create_loader, prepare_batch

    test_dataset = VoxtralDataset(config)
    test_loader = create_loader(test_dataset, config)
    device = utils.get_device()
    ema_model = ema_model.to(device)
    tokenizer = VoxtralTokenizer(config.voxtral_tokenizer_config).to(
        device, torch.float16
    )

    total_batches = max(
        config.test_size // (test_loader.batch_size) // config.world_size, 1
    )

    generations = []

    for n, batch in enumerate(
        tqdm.tqdm(
            itertools.islice(test_loader, total_batches),
            total=total_batches,
            colour="magenta",
            desc="generation_test",
            leave=False,
            disable=config.rank > 0,
        )
    ):
        x = prepare_batch(batch, device=device)

        # Crop the input to the first x tokens
        x_cropped = x[..., : x.size(-1) // 2]

        # Generate new tokens
        generated = ema_model.generate(x_cropped, max_new_tokens=x.size(-1) // 2)

        # Decode the entire sequence
        decoded_generation = tokenizer.decode(generated)

        generations.extend(decoded_generation.unbind())

    log_test_results(generations, step)

    ema_model = ema_model.to("cpu")
    utils.distributed_only(torch.distributed.barrier)()


def log_test_results(
    generations: list[torch.Tensor],
    step: int,
) -> None:
    print("Logging audio...")
    table = wandb.Table(columns=["Generated"])

    for generation in generations:
        # Ensure the tensors are on CPU and convert to numpy arrays
        generation_audio = generation.cpu().numpy().astype(np.float32)

        # Reshape if necessary (assuming the last dimension is time)
        if generation_audio.ndim == 3:
            generation_audio = generation_audio.squeeze(
                1
            )  # Remove the channel dimension if present

        # Ensure the arrays are 2D (channels, time)
        if generation_audio.ndim == 1:
            generation_audio = generation_audio[np.newaxis, :]

        # Transpose to (time, channels) as expected by wandb.Audio
        generation_audio = generation_audio.T

        table.add_data(
            wandb.Audio(generation_audio, sample_rate=24_000),
        )

    wandb.log({"generation_samples": table}, step=step)
