import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetKSoft(nn.Module):
    def __init__(self, backbone: nn.Module, with_distractor = False):
        super().__init__()
        self.backbone = backbone

    def forward(self, support: dict, support_unlabeled: dict, query: dict, num_soft_kmeans_iterations=1):
        support["embeddings"] = self.backbone(support["audio"])
        support_unlabeled["embeddings"] = self.backbone(support_unlabeled["audio"])
        query["embeddings"] = self.backbone(query["audio"])

        support_embeddings = []
        for idx in range(len(support["classlist"])):
            embeddings = support["embeddings"][support["target"] == idx]
            support_embeddings.append(embeddings)
        support_embeddings = torch.stack(support_embeddings)

        prototypes = support_embeddings.mean(dim=1)
        support["prototypes"] = prototypes

        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        distances = distances ** 2
        logits = -distances

        # Soft k-means refinement
        refined_prototypes = self.soft_kmeans(support, support_unlabeled, prototypes)
        support["prototypes"] = refined_prototypes

        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            refined_prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        distances = distances ** 2
        logits = -distances

        # return the logits
        return logits

    def soft_kmeans(self, support_labeled, support_unlabeled, prototypes, num_iterations=1):
      prob_train = []

      for idx in range(len(support_labeled["classlist"])):
            prob = support_labeled["target"] == idx
            prob_train.append(prob.type(torch.int))
      prob_train = torch.stack(prob_train)
      prob_train = prob_train.t()

      support_embeddings = torch.cat([support_labeled["embeddings"], support_unlabeled["embeddings"]], dim=0)
      for _ in range(num_iterations):
          distances = torch.cdist(
              support_unlabeled["embeddings"].unsqueeze(0),
              prototypes.unsqueeze(0),
              p=2
          ).squeeze(0)

          distances = distances ** 2
          soft_assignments_unlabeled = F.softmax(-distances, dim=1)

          prob_all = torch.cat([prob_train, soft_assignments_unlabeled], dim=0)
          prob_all = prob_all.detach()

          refined_prototypes = torch.matmul(prob_all.T, support_embeddings)
          prototypes = refined_prototypes / (prob_all.sum(dim=0, keepdim=True).T + 1e-10)

      return prototypes
