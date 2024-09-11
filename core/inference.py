import os
import yaml
import torch
import logging
from typing import Union

import sys
sys.path.append(os.getcwd())

from core.utils import DataProcessor


class Inference:
    """
    Class for ml/nn model inference
    """
    def __init__(self, cfg: Union[str, dict] = None, logger: logging.Logger = None):
        if isinstance(cfg, dict):
            config = cfg
        else:
            with open(cfg, 'r', encoding="utf-8") as f:
                config = yaml.safe_load(f)

        self.logger = logger

        self.target_sr = config["general"]["sample_rate"]
        self.target_length = config["general"]["audio_length"]

        self.device = torch.device(config["general"]["device"] if torch.cuda.is_available() else "cpu")

        self.dataprocessor = DataProcessor(target_sr=self.target_sr, target_length=self.target_length)

        model_path = config["general"]["inference_model_path"]

        assert os.path.exists(model_path), f"Can`t find the model at the path {model_path}"

        self.nn_model = torch.load(model_path, map_location=self.device).to(self.device)
        self.nn_model.eval()

        self.logger.info(f"Model instantiated to: {self.device}")

    def get_prediction_from_file(
            self,
            filepath: str = None,
            return_positive_probability: bool = False
    ):
        """
        Predictions generation method. Batch size > 1 not supported.
        :param filepath: Path to file
        :param return_positive_probability: Wheter to return positive class probability
        """
        try:
            self.logger.info("Receiving NN predictions from file")

            samples = self.dataprocessor.read_file(filepath, train_mode=False)
            with torch.no_grad():
                samples = samples.to(self.device)
                predictions = self.nn_model(samples).detach().cpu().numpy()

            return round(predictions[0][-1], 3) if return_positive_probability else predictions

        except Exception as e:
            self.logger.exception(str(e))


if __name__ == "__main__":
    level_name = logging.getLevelName("INFO")
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    main_logger = logging.getLogger("USC Inference")
    main_logger.setLevel(logging.getLevelName(level_name))

    test_data_path = r"C:\Users\Che\Desktop\mono.wav"

    inference = Inference("config.yaml", main_logger)

    print(f"From filepath: {inference.get_prediction_from_file(test_data_path)}")
