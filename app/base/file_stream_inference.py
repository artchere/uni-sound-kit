import os
import sys
import yaml
import torch
import logging
import numpy as np
from typing import Union
from app.base.io_stream import FileStream

sys.path.append(os.getcwd())


class FileStreamInference:
    def __init__(self, cfg: Union[str, dict] = None, logger: logging.Logger = None):
        if isinstance(cfg, dict):
            self.config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        self.logger = logger

        self.sample_rate = self.config["general"]["sample_rate"]
        self.window_size = int(self.config["stream"]["window_size"] * self.sample_rate)
        self.buffer_size = int(self.config["stream"]["buffer_size"] * self.sample_rate)
        self.datatype = np.int16 if self.config["stream"]["use_int16"] else np.float32
        self.device = torch.device(self.config["general"]["device"] if torch.cuda.is_available() else "cpu")
        self.inverted_label_map = {v: k for k, v in self.config["general"]["label_map"].items()}
        self.kw_target = int(self.config["stream"]["keyword_label"])
        self.threshold = float(self.config["stream"]["threshold"])
        self.use_argmax = self.config["stream"]["use_argmax"]
        model_path = self.config["general"]["inference_model_path"]

        assert os.path.exists(model_path), f"Can`t find the model at the path {model_path}"

        self.nn_model = torch.load(model_path, map_location=self.device).to(self.device)
        self.nn_model.eval()

    def start_stream(self, filepath: str = None, return_timestamps: bool = False, return_bytes: bool = False):
        file_stream = FileStream(self.config, path_to_audio=filepath, return_bytes=return_bytes)

        collect_to = int(self.window_size / self.buffer_size)
        result_buffer = [1 for _ in range(collect_to - 1)]
        audio_buffer = [np.zeros(self.buffer_size, dtype=self.datatype) for _ in range(collect_to - 1)]

        n_chunks = file_stream.n_chunks

        i = 0
        for chunk in file_stream:
            audio_buffer.append(chunk)

            if len(audio_buffer) == collect_to:
                samples = np.concatenate(audio_buffer)
                diff = self.window_size - len(samples)
                samples = np.concatenate([samples, np.zeros(diff, dtype=self.datatype)]) if diff > 0 else samples
                with torch.no_grad():
                    samples = torch.from_numpy(samples).to(self.device).unsqueeze(0)
                    predictions = self.nn_model(samples).detach().cpu().numpy()
                    if not self.use_argmax:
                        if predictions.squeeze()[self.kw_target] >= self.threshold:
                            prediction = self.kw_target
                        elif 0.5 <= predictions.squeeze()[self.kw_target] < self.threshold:
                            prediction = int(np.flip(np.argsort(predictions.squeeze()))[1].item())
                        else:
                            prediction = int(predictions.argmax(1).item())
                    else:
                        prediction = int(predictions.argmax(1).item())

                    left_bound = (i - (collect_to - 1)) * self.buffer_size / self.sample_rate
                    right_bound = ((i - (collect_to - 1)) * self.buffer_size / self.sample_rate) + \
                                  (int(self.window_size / self.sample_rate))

                    if (n_chunks - (collect_to - 1)) >= i >= (collect_to - 1):
                        if return_timestamps:
                            print(f"{self.inverted_label_map[prediction]}: {left_bound}-{right_bound} sec")
                        else:
                            print(self.inverted_label_map[prediction])

                torch.cuda.empty_cache()
                result_buffer.append(int(prediction))

                result_buffer.pop(0)
                audio_buffer.pop(0)
                i += 1


if __name__ == "__main__":
    level_name = logging.getLevelName("INFO")
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    main_logger = logging.getLogger("USC File Stream Inference")
    main_logger.setLevel(logging.getLevelName(level_name))

    audio_stream = FileStreamInference("config.yaml", main_logger)
    audio_stream.start_stream("C:/Users/Che/desktop/mono.wav", return_timestamps=True)
