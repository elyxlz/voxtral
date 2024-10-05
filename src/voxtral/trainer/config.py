import pydantic_settings as pyds


class VoxtralTrainConfig(pyds.BaseSettings):
    lr: float
    layer_dropout_factor: int = 1  # no layer dropout
