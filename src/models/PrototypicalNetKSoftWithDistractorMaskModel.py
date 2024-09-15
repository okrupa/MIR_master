import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activations):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(input_dim, hidden_dims[i]))
            if activations[i]:
                layers.append(activations[i]())
            input_dim = hidden_dims[i]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PrototypicalNetKSoftWithDistractorMaskModel(nn.Module):
    def __init__(self, backbone: nn.Module, with_distractor = False):
        super().__init__()
        self.backbone = backbone
        self.init_radius = 100.0 #Initial radius for the distractors
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_features = 5
        n_out = 3
        hdim = [20, n_out]
        act_fn = [nn.Tanh, None]
        self.mlp = MLP(self.n_features, hdim, n_out, act_fn).to(self.device)


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

        refined_prototypes = self.soft_kmeans_with_distractor_mask_model(support, support_unlabeled, prototypes)
        support["prototypes"] = refined_prototypes

        distances = torch.cdist(
            query["embeddings"].unsqueeze(0),
            refined_prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        distances = distances ** 2
        logits = -distances

        return logits

    def soft_kmeans_with_distractor_mask_model(self, support_labeled, support_unlabeled, prototypes, num_iterations=1):
      prob_train = []
      for idx in range(len(support_labeled["classlist"])):
            prob = support_labeled["target"] == idx
            prob_train.append(prob.type(torch.int))
      prob_train = torch.stack(prob_train)
      prob_train = prob_train.t()

      support_embeddings = torch.cat([support_labeled["embeddings"], support_unlabeled["embeddings"]], dim=0)
      mask = None
      protos_1 = prototypes
      protos_2 = support_unlabeled["embeddings"]

      pair_dist = torch.cdist(
            prototypes.unsqueeze(0),
            protos_2.unsqueeze(0),
            p=2
        ).squeeze(0)
      mean_dist = torch.mean(pair_dist, dim=1, keepdim=True)
      pair_dist_normalize = pair_dist / mean_dist

      min_dist = torch.min(pair_dist_normalize, dim=1, keepdim=True).values

      max_dist = torch.max(pair_dist_normalize, dim=1, keepdim=True).values

      mean_dist = torch.mean(pair_dist_normalize, dim=1, keepdim=True)

      var_dist = torch.var(pair_dist_normalize, dim=1, keepdim=True)

      mean_dist += torch.tensor(mean_dist == 0.0, dtype=torch.float32)
      var_dist += torch.tensor(var_dist == 0.0, dtype=torch.float32)

      skew = torch.mean(((pair_dist_normalize - mean_dist)**3) / (torch.sqrt(var_dist)**3), dim=1, keepdim=True)

      kurt = torch.mean(((pair_dist_normalize - mean_dist)**4) / (var_dist**2) - 3, dim=1, keepdim=True)

      concatenated = torch.cat((min_dist, max_dist, var_dist, skew, kurt), dim=1).to(self.device)

      dist_features = concatenated.view(-1, self.n_features)

      dist_features = dist_features.to(self.device)

      thresh = self.mlp(dist_features.float())

      scale = torch.exp(thresh[:, 2])
      bias_start = torch.exp(thresh[:, 0])
      bias_add = thresh[:, 1]

      bias_start = bias_start.unsqueeze(0)

      bias_add = bias_add.unsqueeze(0)

      for i in range(num_iterations):
        protos_1 = prototypes
        protos_2 = support_unlabeled["embeddings"]
        pair_dist = torch.cdist(
            protos_1.unsqueeze(0),
            protos_2.unsqueeze(0),
            p=2
        ).squeeze(0)

        m_dist = torch.mean(pair_dist, dim=1)

        m_dist_1 = m_dist.unsqueeze(0)

        m_dist_1 += torch.eq(m_dist_1, 0.0).float()
        if num_iterations > 1:
          bias_i = bias_start + (i / float(num_iterations - 1)) * bias_add
        else:
          bias_i = bias_start

        distances = torch.cdist(
            support_unlabeled["embeddings"].unsqueeze(0),
            prototypes.unsqueeze(0),
            p=2
        ).squeeze(0)

        logits = -distances ** 2

        mask = torch.sigmoid((logits / m_dist_1 + bias_i) * scale)
        logits_shape = logits.shape
        ndata = logits_shape[0]
        ncluster = logits_shape[1]
        logits = logits.view(-1, ncluster)
        prob_unlabel = torch.nn.functional.softmax(logits, dim=-1)
        prob_unlabel_with_mask = prob_unlabel.view(ndata, ncluster) * mask

        prob_all = torch.cat([prob_train, prob_unlabel_with_mask], dim=0)

        if support_unlabeled["embeddings"].shape[0] > 0:
          refined_prototypes = torch.matmul(prob_all.T, support_embeddings)
          prototypes = refined_prototypes / (prob_all.sum(dim=0, keepdim=True).T + 1e-10)

      return prototypes