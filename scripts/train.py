from voxtral.trainer.trainer import VoxtralTrainConfig, train
from voxtral.tokenizer.model import VoxtralTokenizerConfig

config = VoxtralTrainConfig(
    name="voxtral-train-run",
    seed=42,
    mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
    mistral_kwargs={},
    voxtral_tokenizer_config=VoxtralTokenizerConfig(),
    new_vocab_size=2**16,
    lora_rank=64,
    prune_layers=None,  # half parameter count
    codec_hz=55,
    ## ema
    ema_gamma=16,
    ema_every=64,
    ## dataset
    data_path="./data/tokens",
    fake=False,
    overfit=16,
    batch_size=32,
    num_workers=20,
    test_size=16,
    ## speed
    compile=True,
    gradient_checkpointing=True,
    ## opt
    lr=3e-4,
    weight_decay=0.1,
    lr_eps=1e-9,
    lr_betas=(0.9, 0.95),
    grad_norm=1.0,
    warmup_steps=50,
    max_steps=30_000,
    ## test
    test_every=100,
    ## logging and checkpointing
    watch_every=None,
    ckpt_path=None,
    save_every=500,
    push_every=500,
    wandb_project_name="voxtral",
)
train(config)
