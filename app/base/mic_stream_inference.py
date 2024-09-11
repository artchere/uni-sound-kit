import os
import yaml
import torch
import logging
import pyaudio
import numpy as np
from queue import Queue
from typing import Union
import matplotlib.pyplot as plt

import sys
sys.path.append(os.getcwd())


class MicStreamInference:
    """
    Class for predicting streaming data. Heavily adapted from the implementation:
    """
    def __init__(self, cfg: Union[str, dict] = None, logger: logging.Logger = None):
        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.logger = logger

        self.mic_device_id = config["stream"]["device_id"]
        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")

        model_path = config["general"]["inference_model_path"]

        assert os.path.exists(model_path), f"Can`t find the model at the path {model_path}"

        self.nn_model = torch.load(model_path, map_location=self.device).to(self.device)
        self.nn_model.eval()

        self.logger.info(f"Model instantiated to: {self.device}")

        # Recording parameters
        self.datatype = pyaudio.paInt16 if config["stream"]["use_int16"] else pyaudio.paFloat32
        self.target_sr = config["general"]["sample_rate"]
        self.chunk_samples = int(self.target_sr * config["stream"]["buffer_size"])
        self.window_samples = int(self.target_sr * config["stream"]["window_size"])
        self.silence_threshold = 100

        # Data structures and buffers
        self.queue = Queue()
        self.data = np.zeros(self.window_samples, dtype="float32")

        # Plotting parameters
        self.inverted_label_map = {v: k for k, v in config["general"]["label_map"].items()}
        self.kw_target = int(config["stream"]["keyword_label"])
        self.threshold = float(config["stream"]["threshold"])
        self.use_argmax = config["stream"]["use_argmax"]
        self.change_bkg_frames = 2
        self.change_bkg_counter = 0
        self.change_bkg = False

    def start_stream(self, visualize=False):
        """
        Start audio data streaming from microphone
        :return: None
        """
        stream = pyaudio.PyAudio().open(
            input_device_index=self.mic_device_id,
            format=self.datatype,
            channels=1,
            rate=self.target_sr,
            input=True,
            frames_per_buffer=self.chunk_samples,
            stream_callback=self.callback,
        )

        stream.start_stream()

        collect_to = int(self.window_samples / self.chunk_samples)
        result_buffer = [1 for _ in range(collect_to)]
        try:
            while True:
                samples = self.queue.get()
                samples = torch.from_numpy(samples)

                with torch.no_grad():
                    samples = samples.to(self.device).unsqueeze(0)
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

                torch.cuda.empty_cache()
                result_buffer.append(prediction)

                if len(result_buffer) > collect_to:
                    result_buffer.pop(0)

                if visualize:
                    self.plotter(samples.squeeze(), prediction)

                self.logger.info(self.inverted_label_map[prediction])

        except (KeyboardInterrupt, SystemExit):
            stream.stop_stream()
            stream.close()

    def callback(self, in_data, frame_count, time_info, status):
        """
        Obtain the data from buffer and load it to queue
        :param in_data: Daa buffer
        :param frame_count: Frame count
        :param time_info: Time information
        :param status: Status
        """
        data0 = np.frombuffer(in_data, dtype=np.float32)

        if np.abs(data0).mean() < self.silence_threshold:
            self.logger.info(".")
        else:
            self.logger.info("-")

        self.data = np.append(self.data, data0)

        if len(self.data) > self.window_samples:
            self.data = self.data[-self.window_samples:]
            self.queue.put(self.data)

        return in_data, pyaudio.paContinue

    def plotter(self, raw_data, prediction):
        """
        Plot waveform, filterbank energies and hotword presence
        :param raw_data: Audio data array
        :param prediction: Predicted label
        """
        plt.clf()

        # Wave
        plt.subplot(311)
        plt.plot(raw_data[-len(raw_data) // 2:])
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.ylabel("Amplitude")

        # Hotword detection
        plt.subplot(312)
        ax = plt.gca()

        if prediction == self.kw_target:
            self.change_bkg = True

        if self.change_bkg and self.change_bkg_counter < self.change_bkg_frames:
            ax.set_facecolor("lightgreen")

            ax.text(
                x=0.5,
                y=0.5,
                s="KeyWord!",
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=30,
                color="red",
                fontweight="bold",
                transform=ax.transAxes,
            )

            self.change_bkg_counter += 1
        else:
            ax.set_facecolor("salmon")
            self.change_bkg = False
            self.change_bkg_counter = 0

        plt.tight_layout()
        plt.pause(0.01)


if __name__ == "__main__":
    level_name = logging.getLevelName("INFO")
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    main_logger = logging.getLogger("USC Mic Stream Inference")
    main_logger.setLevel(logging.getLevelName(level_name))

    audio_stream = MicStreamInference("config.yaml", main_logger)
    audio_stream.start_stream(visualize=False)
