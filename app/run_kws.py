import os
import sys
import uvicorn

sys.path.append(os.getcwd())
from app.config import app_config


if __name__ == "__main__":
    # Запускаем сервер Uvicorn с приложением FastAPI
    uvicorn.run(
        "app.application:app",  # Путь к приложению FastAPI: "пакет.модуль:экземпляр приложения"
        host=app_config.app_host,  # Читаем хост из конфигурации
        port=app_config.app_port,  # Читаем порт из конфигурации
        log_level=app_config.log_level,  # Устанавливаем уровень логирования
    )
