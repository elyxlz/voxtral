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

    prompts, generations = [], []

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
        x_cropped = x[..., : config.test_num_prompt_tokens]

        # Generate new tokens
        generated = ema_model.generate(
            x_cropped, max_new_tokens=config.test_num_new_tokens
        )
        generated = x + 1  # TODO:

        # Decode the entire sequence
        decoded_prompt = tokenizer.decode(x_cropped)
        decoded_generation = tokenizer.decode(generated)

        prompts.append(decoded_prompt)
        generations.append(decoded_generation)

    log_test_results(prompts, generations, step)

    ema_model = ema_model.to("cpu")
    utils.distributed_only(torch.distributed.barrier)()


def log_test_results(
    prompts: list[torch.Tensor],
    generations: list[torch.Tensor],
    step: int,
) -> None:
    print("Logging audio...")
    table = wandb.Table(columns=["Prompt", "Generated"])

    for prompt, generation in zip(prompts, generations):
        # Ensure the tensors are on CPU and convert to numpy arrays
        prompt_audio = prompt.cpu().numpy().astype(np.float32)
        generation_audio = generation.cpu().numpy().astype(np.float32)

        # Reshape if necessary (assuming the last dimension is time)
        if prompt_audio.ndim == 3:
            prompt_audio = prompt_audio.squeeze(
                1
            )  # Remove the channel dimension if present
        if generation_audio.ndim == 3:
            generation_audio = generation_audio.squeeze(
                1
            )  # Remove the channel dimension if present

        # Ensure the arrays are 2D (channels, time)
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio[np.newaxis, :]
        if generation_audio.ndim == 1:
            generation_audio = generation_audio[np.newaxis, :]

        # Transpose to (time, channels) as expected by wandb.Audio
        prompt_audio = prompt_audio.T
        generation_audio = generation_audio.T

        table.add_data(
            wandb.Audio(prompt_audio, sample_rate=24_000),
            wandb.Audio(generation_audio, sample_rate=24_000),
        )

    wandb.log({"generation_samples": table}, step=step)
