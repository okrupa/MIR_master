import argparse
import logging
import yaml
import torch
import uuid
import csv

from torch import nn
from torch.utils.data import DataLoader
from music_fsl.backbone import Backbone
from torchmetrics import Accuracy
from music_fsl.util import dim_reduce, embedding_plot, batch_device

from models.MUSIC import MUSIC
from models.MUSICWithDistractor import MUSICWithDistractor
from datasets.PredictDataset import PredictDatasetLoader, PredictDataset
from datasets.CustomDataset import CustomDatasetLoader, CustomDataset
from config.ModelConfig import ModelConfig
from config.dataset_type import DatasetType
from config.model_type import PrototypicalNetType, FewShotLearnerType
from datasets.EpisodeDatasetUnlabeled import EpisodeDatasetUnlabeled
from datasets.EpisodeDatasetUnlabeledWithDistractor import EpisodeDatasetUnlabeledWithDistractor


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def choose_model(model_config, checkpoint_path):
    with_distractor = False
    if model_config.n_distractor > 0:
        with_distractor = True

    with_distractor_class = False
    if model_config.model_type == 'musicdistractor':
        with_distractor_class = True
    
    backbone = Backbone(sample_rate=model_config.sample_rate)
    protonet_type = getattr(PrototypicalNetType, model_config.model_type).value
    protonet = protonet_type(backbone, with_distractor = with_distractor_class)
    
    if model_config.model_type == 'music' or model_config.model_type == 'musicdistractor':
        choosen_few_shot_type = model_config.model_type 
    else:
        choosen_few_shot_type = 'kmeans'
    few_shot_type = getattr(FewShotLearnerType, choosen_few_shot_type).value
    learner = few_shot_type.load_from_checkpoint(checkpoint_path, protonet=protonet, with_distractor = with_distractor)
    learner.eval()
    learner = learner.to(DEVICE)
    return learner


def train(learner, data_loader, metric, n_episodes, model_config, with_distractor = False):
    optimizer = torch.optim.Adam(learner.parameters(), lr=learner.learning_rate)
    for episode_idx in range(n_episodes):
        if not with_distractor or model_config.dataset_type == 'custom':
            support, unlabeled, query = data_loader[episode_idx]
        else:
            support, unlabeled, query, non_distractor = data_loader[episode_idx]

        optimizer.zero_grad()

        batch_device(support, DEVICE)
        batch_device(query, DEVICE)
        batch_device(unlabeled, DEVICE)

        if isinstance(learner, MUSIC) or isinstance(learner, MUSICWithDistractor):
            logits = learner.predict_with_unlabeled(support, unlabeled, query)
        else:
            logits = learner.protonet(support, unlabeled, query)

        acc = metric(logits, query["target"])

        loss = learner.loss(logits, query["target"])

        loss.backward()
        optimizer.step()
        print(f"Episode {episode_idx} // Accuracy: {acc.item():.2f} // Loss: {loss}")


def predict(data_loader, predict_loader, learner, model_config, with_distractor):
    sig = nn.Sigmoid()
    if not with_distractor or model_config.dataset_type == 'custom':
            support, unlabeled, query = data_loader[model_config.n_train_episodes-1]
    else:
        support, unlabeled, query, non_distractor = data_loader[model_config.n_train_episodes-1]

    query = predict_loader[model_config.n_train_episodes-1]
    batch_device(support, DEVICE)
    batch_device(unlabeled, DEVICE)
    batch_device(query, DEVICE)

    if model_config.model_type == 'music' or model_config.model_type == 'musicdistractor':
        logits = learner.predict_with_unlabeled(support, unlabeled, query)
    else:    
        logits = learner.protonet(support, unlabeled, query)

    names = query["name"]

    outputs = sig(logits)
    _, predicted = torch.max(outputs, 1)
    path = f'.//outputs/{str(uuid.uuid4())}.csv'
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['file', 'classes'])
        classlist = support["classlist"]
        for name, prediction in zip (names, predicted):
            writer.writerow([name, classlist[prediction]])
    
    logger = logging.getLogger(__name__)
    logger.info(f"Result saved in {path}.")
        
def get_episode_dataset(model_config, data, n_query):
    if model_config.dataset_type == 'custom':
        episodes = CustomDatasetLoader(
            dataset=data,
            n_way=model_config.n_way,
            n_support=model_config.n_support,
            n_query=n_query,
            n_unlabeled = model_config.n_unlabeled
        )
        return episodes
    if model_config.n_distractor == 0:
        episodes = EpisodeDatasetUnlabeled(
            dataset=data,
            n_way=model_config.n_way,
            n_support=model_config.n_support,
            n_query=n_query,
            n_unlabeled = model_config.n_unlabeled,
            n_episodes = model_config.n_train_episodes
        )

    else:
        episodes = EpisodeDatasetUnlabeledWithDistractor(
            dataset=data,
            n_way=model_config.n_way,
            n_support=model_config.n_support,
            n_query=n_query,
            n_unlabeled = model_config.n_unlabeled,
            n_distractor = model_config.n_distractor,
            n_unlabeled_distractor = model_config.n_unlabeled_distractor,
            n_episodes = model_config.n_train_episodes
        )

    return episodes

def main():
    parser = argparse.ArgumentParser(description="Train aruments")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    config = vars(args)

    with open(args.config_path, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)

    model_config = ModelConfig(**config_data)

    dataset = getattr(DatasetType, model_config.dataset_type).value
    
    if model_config.dataset_type == 'custom':
        data = dataset(
        classes=model_config.classes, 
        duration = model_config.duration,
        sample_rate=model_config.sample_rate,
        dataset_path = model_config.dataset_path
    )
    else:
        data = dataset(
                classes=dataset.TRAIN_CLASSES, 
                duration = model_config.duration,
                sample_rate=model_config.sample_rate,
                dataset_path = model_config.dataset_path
            )

    predict_data = PredictDataset(
        duration = model_config.duration,
        sample_rate=model_config.sample_rate,
        dataset_path = model_config.predict_path
    )

    with_distractor = False
    if model_config.n_distractor > 0:
        with_distractor = True

    learner = choose_model(model_config, args.model_path)

    if model_config.n_train_dataset > 0:
        data_loader = get_episode_dataset(model_config, data, model_config.n_query)

        metric = Accuracy(num_classes=model_config.n_way, average="samples")

        train(learner, data_loader, metric, model_config.n_train_dataset, model_config, with_distractor)

    data_loader = get_episode_dataset(model_config, data, 0)

    predict_loader = PredictDatasetLoader(
        dataset=predict_data,
    )

    predict(data_loader, predict_loader, learner, model_config, with_distractor)

if __name__ == "__main__":
   main()