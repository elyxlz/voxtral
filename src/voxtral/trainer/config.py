import pydantic_settings as pyds
import wandb.util
from voxtral.tokenizer.model import VoxtralTokenizerConfig


class BaseConfig(pyds.BaseSettings):
    """Pydantic base settings but env variables take absolute priority"""

    model_config = pyds.SettingsConfigDict(env_parse_none_str="None")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pyds.BaseSettings],
        init_settings: pyds.PydanticBaseSettingsSource,
        env_settings: pyds.PydanticBaseSettingsSource,
        dotenv_settings: pyds.PydanticBaseSettingsSource,
        file_secret_settings: pyds.PydanticBaseSettingsSource,
    ) -> tuple[pyds.PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )  # switched order


class VoxtralTrainConfig(BaseConfig):
    run_id: str = wandb.util.generate_id()

    seed: int = 42
    name: str = "voxtral-test"

    mistral_pretrained_path: str = "nilq/mistral-1L-tiny"  # "mistralai/Mistral-7B-v0.3"
    mistral_kwargs: dict = {}
    voxtral_tokenizer_config: VoxtralTokenizerConfig = VoxtralTokenizerConfig()
    new_vocab_size: int = 2**16
    loss_weights: list[int] = [100, 10, 1]  # text, semantic, acoustic
    lora_rank: int | None = None
    prune_layers: int | None = None  # no layer dropout
    codec_hz: int = 55

    ## ema
    ema_gamma: float = 16
    ema_every: int = 1024

    ## dataset
    data_path: str = "./data/tokens"
    fake: bool = True
    overfit: int | None = None
    batch_size: int = 2
    num_workers: int = 4
    test_size: int = 4

    ## speed
    compile: bool = False
    gradient_checkpointing: bool = False

    ## opt
    lr: float = 3e-4
    weight_decay: float = 0.1
    lr_eps: float = 1e-9
    lr_betas: tuple[float, float] = (0.9, 0.95)
    grad_norm: float = 1.0
    warmup_steps: int = 100
    max_steps: int = 500

    ## test
    test_every: int | None = 10
    generate_kwargs: dict = {}

    ## logging and checkpointing
    watch_every: int | None = None
    ckpt_path: str | None = None
    save_every: int | None = None
    push_every: int | None = None
    wandb_project_name: str = "voxtral"

    ## dist (picked up by env)
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
