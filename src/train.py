import logging
import argparse
import yaml
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from music_fsl.backbone import Backbone
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import SimpleProfiler

from config.ModelConfig import ModelConfig
from config.dataset_type import DatasetType
from config.model_type import PrototypicalNetType, FewShotLearnerType
from datasets.EpisodeDatasetUnlabeled import EpisodeDatasetUnlabeled
from datasets.EpisodeDatasetUnlabeledWithDistractor import EpisodeDatasetUnlabeledWithDistractor
from datasets.CustomDataset import CustomDatasetLoader, CustomDataset


def get_episode_dataset(model_config, train_data, val_data):
    if model_config.dataset_type == 'custom':
        train_episodes = CustomDatasetLoader(
            dataset=train_data,
            n_way=model_config.n_way,
            n_support=model_config.n_support,
            n_query=model_config.n_query,
            n_unlabeled = model_config.n_unlabeled,
            n_episodes = model_config.n_train_episodes
        )

        val_episodes = CustomDatasetLoader(
            dataset=val_data,
            n_way=model_config.n_way,
            n_support=model_config.n_support,
            n_query=model_config.n_query,
            n_unlabeled = model_config.n_unlabeled,
            n_episodes=5
        )

    else:
        if model_config.n_distractor == 0:
            train_episodes = EpisodeDatasetUnlabeled(
                dataset=train_data,
                n_way=model_config.n_way,
                n_support=model_config.n_support,
                n_query=model_config.n_query,
                n_unlabeled = model_config.n_unlabeled,
                n_episodes = model_config.n_train_episodes
            )

            val_episodes = EpisodeDatasetUnlabeled(
                dataset=val_data,
                n_way=model_config.n_way,
                n_support=model_config.n_support,
                n_query=model_config.n_query,
                n_unlabeled = model_config.n_unlabeled,
                n_episodes=5
            )
        else:
            train_episodes = EpisodeDatasetUnlabeledWithDistractor(
                dataset=train_data,
                n_way=model_config.n_way,
                n_support=model_config.n_support,
                n_query=model_config.n_query,
                n_unlabeled = model_config.n_unlabeled,
                n_distractor = model_config.n_distractor,
                n_unlabeled_distractor = model_config.n_unlabeled_distractor,
                n_episodes = model_config.n_train_episodes
            )

            val_episodes = EpisodeDatasetUnlabeledWithDistractor(
                dataset=val_data,
                n_way=model_config.n_way,
                n_support=model_config.n_support,
                n_query=model_config.n_query,
                n_unlabeled = model_config.n_unlabeled,
                n_distractor = model_config.n_distractor,
                n_unlabeled_distractor = model_config.n_unlabeled_distractor,
                n_episodes=5
            )
    return train_episodes, val_episodes

def choose_model(model_config ):
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
    learner = few_shot_type(protonet, with_distractor = with_distractor)
    return learner

def main():
    parser = argparse.ArgumentParser(description="Train aruments")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--model_output_path", required=True)
    args = parser.parse_args()
    config = vars(args)


    with open(args.config_path, 'r') as yaml_file:
        config_data = yaml.safe_load(yaml_file)


    model_config = ModelConfig(**config_data)
    logger = logging.getLogger(__name__)

    dataset = getattr(DatasetType, model_config.dataset_type).value

    if model_config.dataset_type == 'custom':
        train_data = dataset(
            classes=model_config.classes, 
            duration = model_config.duration,
            sample_rate=model_config.sample_rate,
            dataset_path = model_config.dataset_path
        )

        val_data = dataset(
            classes=model_config.classes,
            duration = model_config.duration,
            sample_rate=model_config.sample_rate,
            dataset_path = model_config.dataset_path
        )
    else:
        train_data = dataset(
            classes=dataset.TRAIN_CLASSES, 
            duration = model_config.duration,
            sample_rate = model_config.sample_rate,
            dataset_path = model_config.dataset_path
        )

        val_data = dataset(
            classes=dataset.TEST_CLASSES,
            duration = model_config.duration,
            sample_rate=model_config.sample_rate,
            dataset_path = model_config.dataset_path
        )

    
    train_episodes, val_episodes = get_episode_dataset(model_config, train_data, val_data)

    train_loader = DataLoader(
        train_episodes,
        batch_size=None,
        num_workers=model_config.num_workers
    )

    val_loader = DataLoader(
        val_episodes,
        batch_size=None,
        num_workers=model_config.num_workers
    )

    learner = choose_model(model_config)


    logger.info(f"Started training {config_data['model_type']} model on dataset {config_data['dataset_type']}.")

    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=model_config.log_every_n_steps,
        val_check_interval=model_config.val_check_interval,
        profiler=SimpleProfiler(
            filename="profile.txt",
        ),
        logger=TensorBoardLogger(
            save_dir=args.model_output_path,
            name="logs"
        ),
    )

    trainer.fit(learner, train_loader, val_dataloaders=val_loader)
    logger.info(f"Training ended. Model can by found in {args.model_output_path}.")


if __name__ == "__main__":
   main()