import os
import sys
from fastapi import FastAPI
from functools import lru_cache

sys.path.append(os.getcwd())
from app.routers import kws, misc


@lru_cache
def get_application() -> FastAPI:

    application = FastAPI()

    # Подключение роутеров
    application.include_router(kws.router)
    application.include_router(misc.router)

    return application


app = get_application()
