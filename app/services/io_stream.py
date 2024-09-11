import librosa
from io import BytesIO
from typing import Union, Any


class FileStream:
    def __init__(self, cfg: Union[dict, Any], path_to_audio: str = None, audio_data: BytesIO = None,
                 return_bytes: bool = False):
        if isinstance(cfg, dict):
            config = cfg
        else:
            # Если cfg — объект Pydantic, то мы просто берем dict из него
            config = cfg.dict()

        sample_rate = config["general"]["sample_rate"]
        buffer_size = config["stream"]["buffer_size"]

        if path_to_audio:
            # Загрузка аудио из файла
            self.audio, _ = librosa.load(path_to_audio, sr=sample_rate, mono=True)
        elif audio_data:
            # Загрузка аудио из байтового потока
            self.audio, _ = librosa.load(audio_data, sr=sample_rate, mono=True)
        else:
            raise ValueError("Either 'path_to_audio' or 'audio_data' must be provided")

        self.i = 0
        self.chunk_size = seconds2samples(buffer_size, sample_rate)
        self.n_chunks = (len(self.audio) + self.chunk_size - 1) // self.chunk_size
        self.return_bytes = return_bytes
        self.eos = len(self.audio) == 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.eos:
            raise StopIteration

        start = self.i * self.chunk_size
        end = (self.i + 1) * self.chunk_size
        frame = self.audio[start:end]

        self.i += 1

        if self.i >= self.n_chunks:
            self.eos = True

        return frame.tobytes() if self.return_bytes else frame

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.eos:
            raise StopAsyncIteration

        return self.__next__()


def seconds2samples(sec, sample_rate):
    return int(sec * sample_rate)
