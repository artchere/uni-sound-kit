import os
import json
import yaml
import logging
from tqdm import tqdm

import torch
import numpy as np
from torch import nn
from sklearn import metrics
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.getcwd())

from core.nn_modules.extractor import FeatureExtractor
from core.nn_modules.classifier import LinearHead
from core.nn_modules.cnnt import CNNT
from core.nn_modules.ecapa_tdnn import ECAPATDNN
from core.nn_modules.loss import F1Loss
from core.nn_modules.m5net import M5Net
from core.nn_modules.matchbox import MatchboxNet
from core.nn_modules.soundnet import SoundNet
from core.nn_modules.speaknet import SpeakNet
from core.nn_modules.tc_resnet import TCResNet14
from core.nn_modules.uit import UITBase
from core.nn_modules.voxseg import VoxSeg
from core.utils import PDDataset, test_memory_requirement, plot_history, plot_confusion_matrix


class Trainer:
    def __init__(self, cfg_path: str = None, logger: logging.Logger = None):
        self.logger = logger
        self.logger.info("Configuration file reading")

        with open(cfg_path, 'r', encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        os.makedirs(self.config["general"]["path"], exist_ok=True)
        os.makedirs(self.config["general"]["ckpt_path"], exist_ok=True)

        self.device = self.config["general"]["device"] if torch.cuda.is_available() else "cpu"
        self.base_models = {
            "cnnt": CNNT,
            "ecapa": ECAPATDNN,
            "m5net": M5Net,
            "matchbox": MatchboxNet,
            "soundnet": SoundNet,
            "speaknet": SpeakNet,
            "tc-resnet": TCResNet14,
            "uit": UITBase,
            "voxseg": VoxSeg
        }
        self.head = LinearHead

        self.task_name = self.config["train"]["task_name"]

        self.logger.info(f"Computing device: {self.device}")

    def nn_train_val_pipeline(
            self,
            model: torch.nn.Module = None,
            train_dataloader: torch.utils.data.DataLoader = None,
            val_dataloader: torch.utils.data.DataLoader = None,
            test_dataloader: torch.utils.data.DataLoader = None,
            loss_function: any = None,
            optimizer: any = None,
            n_epochs: int = None
    ):
        """
        Low-level method for model training
        :param model: initialized torch nn.Module object
        :param train_dataloader: torch Dataloader object
        :param val_dataloader: torch Dataloader object
        :param test_dataloader: torch Dataloader object
        :param loss_function: initialized torch loss fn object
        :param optimizer: initialized torch optimizer object
        :param n_epochs: nnumber of epochs
        :return: Custom history dictionary with epochs and metrics
        """
        avg_train_total_loss = []
        avg_val_total_loss = []
        avg_val_total_precision = []
        avg_val_total_recall = []
        avg_val_total_f1 = []
        best_val_loss = np.inf
        best_val_metric = 0

        results = {}

        model.to(self.device)

        for epoch in range(n_epochs):
            self.logger.info("")
            self.logger.info(f"Epoch {epoch+1}")

            # Training block
            num_train_batches = len(train_dataloader)
            train_epoch_loss = 0

            model.train()

            for x, y in tqdm(train_dataloader):
                optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = loss_function(output, y)

                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.cpu().item()

            avg_train_loss = train_epoch_loss / num_train_batches
            avg_train_total_loss += [avg_train_loss]

            self.logger.info(f"Train loss: {round(float(avg_train_loss), 4)}")

            # Validation block
            num_val_batches = len(val_dataloader)
            val_epoch_loss = 0
            val_epoch_f1 = 0
            val_epoch_precision = 0
            val_epoch_recall = 0

            model.eval()
            with torch.no_grad():
                for x, y in tqdm(val_dataloader):
                    x = x.to(self.device)
                    y = y.to(self.device)
                    output = model(x)

                    val_epoch_loss += loss_function(output, y).cpu().item()
                    val_epoch_f1 += metrics.f1_score(
                        y.cpu().numpy(),
                        output.argmax(1).cpu().numpy(),
                        average='weighted',
                        zero_division=0
                    )
                    val_epoch_precision += metrics.precision_score(
                        y.cpu().numpy(),
                        output.argmax(1).cpu().numpy(),
                        average='weighted',
                        zero_division=0
                    )
                    val_epoch_recall += metrics.recall_score(
                        y.cpu().numpy(),
                        output.argmax(1).cpu().numpy(),
                        average='weighted',
                        zero_division=0
                    )

            avg_val_loss = val_epoch_loss / num_val_batches
            avg_val_total_loss += [avg_val_loss]
            avg_val_f1 = val_epoch_f1 / num_val_batches
            avg_val_total_f1 += [avg_val_f1]
            avg_val_precision = val_epoch_precision / num_val_batches
            avg_val_total_precision += [avg_val_precision]
            avg_val_recall = val_epoch_recall / num_val_batches
            avg_val_total_recall += [avg_val_recall]

            results[int(epoch+1)] = (avg_train_loss, avg_val_loss, avg_val_precision, avg_val_recall, avg_val_f1)

            self.logger.info("Validation stats:")
            self.logger.info(f"Val loss: {round(avg_val_loss, 4)}")
            self.logger.info(f"Val Precision: {round(avg_val_precision, 2)}")
            self.logger.info(f"Val Recall (Specificity): {round(avg_val_recall, 2)}")
            self.logger.info(f"Val F1-score: {round(avg_val_f1, 2)}")

            model_type = f"{self.config['train']['nn_config']['model_type']}"
            sr = f"_{str(self.config['general']['sample_rate']).strip('000')}khz"
            tl = f"_{self.config['general']['audio_length']}s"
            ex_type = f"_{self.config['train']['nn_config']['extractor']['extractor_type']}" \
                if model_type not in ["m5net", "soundnet"] else ''
            last_path = os.path.join(self.config["general"]["ckpt_path"],
                                     f"{self.task_name}_{model_type}{tl}{sr}{ex_type}_{epoch+1}.pt")
            best_path = os.path.join(self.config["general"]["ckpt_path"],
                                     f"{self.task_name}_{model_type}{tl}{sr}{ex_type}.pt")

            if self.config["train"]["save_last_ckpt"]:
                torch.save(model, last_path)
                self.logger.info("Saved model checkpoint on epoch")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if not self.config["train"]["save_by_metric"]:
                    torch.save(model, best_path)
                    self.logger.info("Saved best model checkpoint by loss value")

            if avg_val_f1 > best_val_metric:
                best_val_metric = avg_val_f1
                if self.config["train"]["save_by_metric"]:
                    torch.save(model, best_path)
                    self.logger.info("Saved best model checkpoint by F1-score value")

            avg_val_total_loss += [avg_val_loss]
            avg_val_total_f1 += [avg_val_f1]

        # Test block
        test_true = []
        test_pred = []
        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_dataloader):
                x = x.to(self.device)
                test_true += [y.cpu().numpy()]
                test_pred += [model(x).argmax(1).cpu().numpy()]
        test_data = (test_pred, test_true)

        self.logger.info("")
        self.logger.info("Training stats:")
        self.logger.info(f"Best val loss: {round(best_val_loss, 4)}")
        self.logger.info(f"Avg val loss on {n_epochs} epochs: {round(float(np.mean(avg_val_total_loss)), 4)}")
        self.logger.info(f"STD val loss on {n_epochs} epochs: {round(float(np.std(avg_val_total_loss)), 2)}")
        self.logger.info(f"Best val F1: {round(best_val_metric, 3)}")
        self.logger.info(f"Avg val F1 on {n_epochs} epochs: {round(float(np.mean(avg_val_total_f1)), 3)}")
        self.logger.info(f"STD val F1 on {n_epochs} epochs: {round(float(np.std(avg_val_total_f1)), 2)}")
        self.logger.info(f"Avg val Precision on {n_epochs} epochs: {round(float(np.mean(avg_val_total_precision)), 3)}")
        self.logger.info(f"Avg val Recall on {n_epochs} epochs: {round(float(np.mean(avg_val_total_recall)), 3)}")

        return results, test_data

    def train(self):
        """
        Hi-level method for model training
        """
        state = 99

        torch.manual_seed(state)
        self.logger.info("NN training start")

        # Data preparation
        train_data_path = self.config["general"]["train_data_path"]
        test_data_path = self.config["general"]["test_data_path"]
        label_map = self.config["general"]["label_map"]
        audio_length = self.config["general"]["audio_length"]
        target_sr = self.config["general"]["sample_rate"]

        paths = []
        labels = []
        for label in os.listdir(train_data_path):
            label_path = os.path.join(train_data_path, label)
            if label in label_map.keys():
                for f in os.listdir(label_path):
                    if f.endswith(".wav"):
                        paths += [os.path.join(label_path, f)]
                        labels += [label_map[label]]

        test_filepaths = []
        test_targets = []
        for label in os.listdir(test_data_path):
            label_path = os.path.join(test_data_path, label)
            if label in label_map.keys():
                for f in os.listdir(label_path):
                    if f.endswith(".wav"):
                        test_filepaths += [os.path.join(label_path, f)]
                        test_targets += [label_map[label]]

        assert len(paths) > 0, "Train data is empty. Make sure that your directory contains wav-files."

        train_filepaths, val_filepaths, train_targets, val_targets = train_test_split(
            paths,
            labels,
            test_size=self.config["train"]["val_size"],
            shuffle=True,
            random_state=state,
            stratify=labels
        )

        # Dataloaders init
        batch_size = self.config["train"]["batch_size"]

        train_dataset = PDDataset((train_filepaths, train_targets), target_sr=target_sr, target_length=audio_length)
        val_dataset = PDDataset((val_filepaths, val_targets), target_sr=target_sr, target_length=audio_length)
        test_dataset = PDDataset((test_filepaths, test_targets), target_sr=target_sr, target_length=audio_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

        test_tensor = torch.randn(2, 1, int(target_sr * audio_length))  # Batch == 2 for trouble-free BN-layer ops

        self.logger.info("Dataloaders created")

        # Extractor unit
        extractor_params = self.config["train"]["nn_config"]["extractor"]
        extractor_params["target_sr"] = self.config["general"]["sample_rate"]
        extractor = FeatureExtractor(**extractor_params)
        extractor.to(self.device)

        # Input shape for model init
        freq_size, time_size = extractor(test_tensor.to(self.device)).squeeze().shape[1:]

        # Model unit
        model_type = self.config["train"]["nn_config"]["model_type"]
        model_params = self.config["train"]["nn_config"]["models"][model_type]
        if model_type == "uit":
            model_params["freq_size"] = freq_size
            model_params["time_size"] = time_size
        if model_type in ["cnnt", "matchbox"]:
            model_params["freq_size"] = freq_size
        base_model = self.base_models[model_type](**model_params) if model_type not in \
            ["soundnet", "voxseg"] \
            else self.base_models[model_type]()
        base_model.to(self.device)

        # Input shape for head init
        body = nn.Sequential(extractor, base_model) if model_type not in ["soundnet", "m5net"] else base_model
        body.to(self.device)
        head_input_size = list(body(test_tensor.to(self.device)).squeeze().shape[1:])

        torch.cuda.empty_cache()

        # Head unit
        head_params = self.config["train"]["nn_config"]["head"]
        head = self.head(
            dropout=head_params["dropout"],
            hidden_units=head_input_size,
            n_classes=len(label_map),
            norm_layer=head_params["layer_norm"]
        )
        head.to(self.device)

        # Full model init
        model = nn.Sequential(extractor, base_model, head) if model_type not in ["soundnet", "m5net"] \
            else nn.Sequential(base_model, head)

        try:
            self.logger.info(test_memory_requirement(
                model,
                test_tensor,
                self.device)
            )
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.info(e)
            pass

        epochs = self.config["train"]["epochs"]
        learning_rate = self.config["train"]["learning_rate"]
        criterion = F1Loss(return_cross_entropy=self.config["train"]["use_ce_loss"])
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self.logger.info("Model configured")

        if not head_params["as_classifier"]:
            pass  # TODO: pipeline for unlabeled training (voice identification/verification tasks)

        history = self.nn_train_val_pipeline(
            model,
            train_loader,
            val_loader,
            test_loader,
            criterion,
            opt,
            epochs
        )

        base_path = os.path.join(self.config["general"]["path"], self.task_name)
        os.makedirs(base_path, exist_ok=True)
        cm_path = os.path.join(base_path, "confusion_matrix.png")
        history_path = os.path.join(base_path, "history.png")
        cfg_path = os.path.join(base_path, "config.yaml")

        plot_history(
            history[0],
            history_path
        )
        plot_confusion_matrix(
            history[1][0],
            history[1][1],
            list(self.config["general"]["label_map"].values()),
            list(self.config["general"]["label_map"].keys()),
            cm_path
        )

        with open(cfg_path, 'w', encoding='utf8') as w:
            json.dump(self.config, w, indent=2)

        self.logger.info("")
        self.logger.info("Done training")


if __name__ == "__main__":
    level_name = logging.getLevelName("INFO")
    logging.basicConfig(
        level=level_name,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    main_logger = logging.getLogger("USC Trainer")
    main_logger.setLevel(logging.getLevelName(level_name))

    trainer = Trainer("config.yaml", main_logger)
    trainer.train()
