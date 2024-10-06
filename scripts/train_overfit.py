from voxtral.tokenizer.model import VoxtralTokenizerConfig
from voxtral.trainer.trainer import VoxtralTrainConfig, train

config = VoxtralTrainConfig(
    name="voxtral-train-overfit",
    seed=42,
    mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
    mistral_kwargs={},
    voxtral_tokenizer_config=VoxtralTokenizerConfig(),
    new_vocab_size=2**16,
    lora_rank=None,
    prune_layers=2,  # half parameter count
    codec_hz=55,
    ## ema
    ema_gamma=16,
    ema_every=256,
    ## dataset
    data_path="./data/tokens",
    fake=False,
    overfit=16,
    batch_size=8,
    num_workers=20,
    test_size=16,
    ## speed
    compile=True,
    gradient_checkpointing=True,
    ## opt
    lr=1e-4,
    weight_decay=0.1,
    lr_eps=1e-9,
    lr_betas=(0.9, 0.95),
    grad_norm=1.0,
    warmup_steps=200,
    max_steps=50_000,
    ## test
    test_every=5_000,
    generate_kwargs={"do_sample": False},  # greedy sampling when testing overfit
    ## logging and checkpointing
    watch_every=None,
    ckpt_path=None,
    save_every=5_000,
    push_every=5_000,
    wandb_project_name="voxtral",
)
train(config)
