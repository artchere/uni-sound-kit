import os
import sys
import yaml
from pydantic import BaseModel, Field
from typing import Dict

from pydantic_settings import BaseSettings


sys.path.append(os.getcwd())


# Настройки приложения
class AppConfig(BaseSettings):
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8003, env="APP_PORT")
    log_level: str = Field("info", env="LOG_LEVEL")
    debug_mode: bool = Field(True, env="DEBUG_MODE")

    class Config:
        env_file = ".env"


# Настройки модели
class GeneralConfig(BaseModel):
    log_level: str
    path: str
    inference_model_path: str
    device: str
    audio_length: int
    sample_rate: int
    label_map: Dict[str, int]


class StreamConfig(BaseModel):
    device_id: int
    use_int16: bool
    keyword_label: int
    buffer_size: float
    window_size: float
    threshold: float
    use_argmax: bool


class ModelConfig(BaseModel):
    general: GeneralConfig
    stream: StreamConfig


# Функция для загрузки конфигурации из YAML-файла
def load_config_from_yaml(yaml_path: str) -> ModelConfig:
    with open(yaml_path, 'r') as file:
        config_data = yaml.safe_load(file)
    return ModelConfig(**config_data)


# Инициализация конфигураций
app_config = AppConfig()
model_config = load_config_from_yaml("config.yaml")
