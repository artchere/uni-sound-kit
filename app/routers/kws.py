import os
import sys
from io import BytesIO
from functools import lru_cache
from fastapi import APIRouter, HTTPException, Depends, Request
from starlette.responses import JSONResponse

sys.path.append(os.getcwd())

from app.services.inference import InferenceService
from app.config import model_config
from app.services.io_stream import FileStream


router = APIRouter()


@lru_cache
def get_inference_service():
    return InferenceService(model_config)


@router.post("/detect-keyword")
async def detect_keyword(request: Request,
                         inference_service: InferenceService = Depends(get_inference_service)):
    # Получаем тело запроса как байты
    audio_bytes = await request.body()

    # Проверяем, что файл не пустой
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio file provided or the file is empty.")

    # Используем `FileStream` с `BytesIO`
    audio_stream = BytesIO(audio_bytes)
    file_stream = FileStream(cfg=model_config, audio_data=audio_stream, return_bytes=False)

    # Выполнение инференса
    keyword_detected = inference_service.infer(file_stream)

    return JSONResponse({"keyword_detected": keyword_detected})
