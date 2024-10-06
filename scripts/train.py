from datetime import datetime
from voxtral.tokenizer.model import VoxtralTokenizerConfig
from voxtral.trainer.trainer import VoxtralTrainConfig, train

config = VoxtralTrainConfig(
    name=f"voxtral-{datetime.now().strftime('%d%m%y')}",
    seed=42,
    mistral_pretrained_path="mistralai/Mistral-7B-v0.3",
    mistral_kwargs={},
    voxtral_tokenizer_config=VoxtralTokenizerConfig(),
    new_vocab_size=2**16,
    loss_weights=[100, 10, 1],  # text, semantic, acoustic
    lora_rank=None,
    prune_layers=2,  # half parameter count
    codec_hz=55,
    ## ema
    ema_gamma=16,
    ema_every=1024,
    ## dataset
    data_path="./data/tokens",
    fake=False,
    overfit=16,
    batch_size=8,
    num_workers=20,
    test_size=32,
    ## speed
    compile=True,
    gradient_checkpointing=True,
    ## opt
    lr=1e-4,
    weight_decay=0.1,
    lr_eps=1e-9,
    lr_betas=(0.9, 0.95),
    grad_norm=1.0,
    warmup_steps=1_000,
    max_steps=500_000,
    ## test
    test_every=10_000,
    generate_kwargs={},
    ## logging and checkpointing
    watch_every=None,
    ckpt_path=None,
    save_every=10_000,
    push_every=10_000,
    wandb_project_name="voxtral",
)
train(config)
