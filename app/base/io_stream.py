import os
import sys
import yaml
import asyncio
import librosa
import pyaudio
import numpy as np
from typing import Union

sys.path.append(os.getcwd())


class MicStream:
    def __init__(self, cfg: Union[str, dict] = None, return_bytes: bool = False):
        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.sample_rate = config["general"]["sample_rate"]
        self.buffer_size = seconds2samples(config["stream"]["buffer_size"], self.sample_rate)
        self.p = pyaudio.PyAudio()

        self.stream = self.p.open(
            input_device_index=config["stream"]["device_id"],
            frames_per_buffer=self.buffer_size,
            format=pyaudio.paInt16 if config["stream"]["use_int16"] else pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True
        )

        self.return_bytes = return_bytes
        self.buffer_dtype = np.int16 if config["stream"]["use_int16"] else np.float32

    def read(self):
        return self.stream.read(self.buffer_size, exception_on_overflow=False) \
            if self.return_bytes else np.frombuffer(
            self.stream.read(self.buffer_size, exception_on_overflow=False), dtype=self.buffer_dtype
        )

    def close(self):
        if hasattr(self, "stream") and self.stream is not None:
            self.stream.close()
            self.stream = None

        if hasattr(self, "p") and self.p is not None:
            self.p.terminate()
            self.p = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


class FileStream:
    def __init__(self, cfg: Union[str, dict] = None, path_to_audio: str = None, return_bytes: bool = False):
        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        sample_rate = config["general"]["sample_rate"]
        buffer_size = config["stream"]["buffer_size"]

        self.audio, _ = librosa.load(path_to_audio, sr=sample_rate, mono=True)

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


def stream_mic(cfg, return_bytes=False):
    with MicStream(cfg, return_bytes=return_bytes) as mic:
        while True:
            yield mic.read()


async def stream_mic_async(cfg, return_bytes=False):
    for frames in stream_mic(cfg, return_bytes=return_bytes):
        if frames is None:
            await asyncio.sleep(1e-3)
            continue

        yield frames


if __name__ == "__main__":
    filepath = "C:/Users/Che/Desktop/mono.wav"

    file_stream = FileStream("config.yaml", path_to_audio=filepath, return_bytes=False)
    mic_stream = stream_mic("config.yaml", return_bytes=False)

    for chunk in file_stream:
        print(chunk)

    for chunk in mic_stream:
        print(chunk)
