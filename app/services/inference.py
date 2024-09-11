import os

import torch
import numpy as np
from app.services.io_stream import FileStream
from app.config import ModelConfig


class InferenceService:
    def __init__(self, config: ModelConfig):
        self.sample_rate = config.general.sample_rate
        self.window_size = int(config.stream.window_size * self.sample_rate)
        self.buffer_size = int(config.stream.buffer_size * self.sample_rate)
        self.datatype = np.int16 if config.stream.use_int16 else np.float32
        self.device = torch.device(config.general.device if torch.cuda.is_available() else "cpu")
        self.model_path = config.general.inference_model_path
        self.kw_target = config.stream.keyword_label
        self.inverted_label_map = {v: k for k, v in config.general.label_map.items()}
        self.threshold = config.stream.threshold
        self.use_argmax = config.stream.use_argmax

        self.model = self.load_model()

    def load_model(self):
        assert os.path.exists(self.model_path), f"Can't find the model at the path {self.model_path}"
        model = torch.load(self.model_path, map_location=self.device).to(self.device)
        model.eval()
        return model

    def infer(self, file_stream: FileStream) -> bool:
        collect_to = int(self.window_size / self.buffer_size)
        result_buffer = [1 for _ in range(collect_to - 1)]
        audio_buffer = [np.zeros(self.buffer_size, dtype=self.datatype) for _ in range(collect_to - 1)]

        keyword_detected = False

        for i, chunk in enumerate(file_stream):
            audio_buffer.append(chunk)

            if len(audio_buffer) == collect_to:
                samples = np.concatenate(audio_buffer)
                diff = self.window_size - len(samples)
                samples = np.concatenate([samples, np.zeros(diff, dtype=self.datatype)]) if diff > 0 else samples
                with torch.no_grad():
                    samples = torch.from_numpy(samples).to(self.device).unsqueeze(0)
                    predictions = self.model(samples).detach().cpu().numpy()
                    if not self.use_argmax:
                        if predictions.squeeze()[self.kw_target] >= self.threshold:
                            prediction = self.kw_target
                        elif 0.5 <= predictions.squeeze()[self.kw_target] < self.threshold:
                            prediction = int(np.flip(np.argsort(predictions.squeeze()))[1].item())
                        else:
                            prediction = int(predictions.argmax(1).item())
                    else:
                        prediction = int(predictions.argmax(1).item())

                    result_buffer.append(int(prediction))
                    result_buffer.pop(0)
                    audio_buffer.pop(0)

                    if result_buffer[-3:] == [self.kw_target, self.kw_target, self.kw_target]:
                        keyword_detected = True
                        break

                torch.cuda.empty_cache()

        return keyword_detected
