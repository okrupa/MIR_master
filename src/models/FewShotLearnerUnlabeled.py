import torch
import pytorch_lightning as pl

from torch import nn
from torchmetrics import Accuracy

class FewShotLearnerUnlabeled(pl.LightningModule):

    def __init__(self,
        protonet: nn.Module,
        learning_rate: float = 1e-4,
        with_distractor = False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.protonet = protonet
        self.learning_rate = learning_rate
        self.with_distractor = with_distractor

        self.loss = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            'accuracy': Accuracy()
        })

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def step(self, batch, batch_idx, tag: str):
        if self.with_distractor:
            support, unlabeled, query, non_distractor = batch
        else:
            support, unlabeled, query = batch

        logits = self.protonet(support, unlabeled, query)
        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")