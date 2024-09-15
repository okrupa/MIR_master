import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalNetKSoftWithDistractor(nn.Module):
    def __init__(self, backbone: nn.Module, with_distractor = False):
        super().__init__()
        self.backbone = backbone
        self.learn_radius= True #Whether or not to learn distractor radius
        self.init_radius = 100.0 #Initial radius for the distractors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_distractor_radius = nn.Parameter(torch.tensor(np.log(self.init_radius), dtype=torch.float32, device=self.device))


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
        refined_prototypes = self.soft_kmeans_with_distractor(support, support_unlabeled, prototypes)
        support["prototypes"] = refined_prototypes

        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            refined_prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        distances = distances ** 2
        logits = -distances

        return logits

    def soft_kmeans_with_distractor(self, support_labeled, support_unlabeled, prototypes, num_iterations=1):
      prob_train = []

      for idx in range(len(support_labeled["classlist"])+1):
            prob = support_labeled["target"] == idx
            prob_train.append(prob.type(torch.int))
      prob_train = torch.stack(prob_train)
      prob_train = prob_train.t()

      # Initialize cluster radii.
      radii = [None] * (len(support_labeled["classlist"]) + 1)
      bsize = 1
      for kk in range(len(support_labeled["classlist"])):
        radii[kk] = torch.ones((bsize, 1), device=self.device) * 1.0


      if self.learn_radius:
        if not self.log_distractor_radius:
          self.log_distractor_radius = torch.nn.Parameter(torch.tensor(np.log(self.init_radius), dtype=torch.float32, device=self.device))

        distractor_radius = torch.exp(self.log_distractor_radius)
      else:
        distractor_radius = self.init_radius
      distractor_radius = distractor_radius if support_unlabeled["embeddings"].shape[1] > 0 else 100000.0

      radii[-1] = torch.ones((bsize, 1), device=self.device) * distractor_radius

      support_embeddings = torch.cat([support_labeled["embeddings"], support_unlabeled["embeddings"]], dim=0)

      prototypes = torch.cat([prototypes, torch.zeros_like(prototypes)[0:1, :]], dim=0)
      for _ in range(num_iterations):
          distances = torch.cdist(
              support_unlabeled["embeddings"].unsqueeze(0),
              prototypes.unsqueeze(0),
              p=2
          ).squeeze(0)

          distances = distances ** 2

          # NOTE in the paper is 1/r^2 and in the code on the git is 2/r^2
          radii = torch.cat(radii, dim=0).T.to(distances.device)
          logits = -distances / 2.0 / (radii**2)
          norm_constant = 0.5 * torch.log(torch.tensor(2*np.pi)) + torch.log(radii)


          logits -= norm_constant

          soft_assignments_unlabeled = F.softmax(logits, dim=1)
          prob_all = torch.cat([prob_train, soft_assignments_unlabeled], dim=0)

          refined_prototypes = torch.matmul(prob_all.T, support_embeddings)
          prototypes = refined_prototypes / (prob_all.sum(dim=0, keepdim=True).T + 1e-10)

      return prototypes