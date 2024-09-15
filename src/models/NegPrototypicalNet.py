import torch
import torch.nn.functional as F
from torch import nn


class NegPrototypicalNet(nn.Module):

    def __init__(self, backbone: nn.Module, with_distractor = False):
        super().__init__()
        self.backbone = backbone
        self.with_distractor = with_distractor

    def get_preds_position(self, unlabel, position, _position, thres=0.2):
        r = []
        un_idx = []
        for idx in range(len(position)):
            pos = position[idx]
            _pos = _position[idx]
            _out = unlabel[idx]
            out = F.softmax(_out)
            if len(pos) == 1:
                un_idx.append(idx)
                continue
            conf = torch.min(out)
            if conf.item() > thres:
                un_idx.append(idx)
                if len(_pos) == 0:
                    r.append(torch.argmin(out).item())
                else:
                    r.append(_pos[-1])
                continue
            t = torch.argmin(out).item()
            a = pos[t]
            _position[idx].append(a)
            position[idx].remove(a)
            r.append(t)
        return torch.tensor(r), un_idx

    def forward(self, support: dict, query = None, support_unlabeled = None, positions = None):
        """
        Forward pass through the protonet.

        Args:
            support (dict): A dictionary containing the support set.
                The support set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
                    - label: A tensor of shape (n_support) with label indices
                    - classlist: A tensor of shape (n_classes) containing the list of classes in this episode
            support_unlabeled (dict): A dictionary containing the unlabeled set.
                The unlabeled set dict must contain the following keys:
                    - audio: A tensor of shape (n_support, n_channels, n_samples)
            query (dict): A dictionary containing the query set.
                The query set dict must contain the following keys:
                    - audio: A tensor of shape (n_query, n_channels, n_samples)

        Returns:
            logits (torch.Tensor): A tensor of shape (n_query, n_classes) containing the logits

        After the forward pass, the support dict is updated with the following keys:
            - embeddings: A tensor of shape (n_support, n_features) containing the embeddings
            - prototypes: A tensor of shape (n_classes, n_features) containing the prototypes

        The query dict is updated with
            - embeddings: A tensor of shape (n_query, n_features) containing the embeddings

        """

        if query and support_unlabeled:
          support["embeddings"] = self.backbone(support["audio"])
          support_unlabeled["embeddings"] = self.backbone(support_unlabeled["audio"])
          query["embeddings"] = self.backbone(query["audio"])

          prototypes = []
          for idx in range(len(support["classlist"])):
              s_embeddings = support["embeddings"][support["target"] == idx]
              u_embeddings = support_unlabeled["embeddings"][support_unlabeled["target"] == idx]
              embeddings = torch.cat((s_embeddings, u_embeddings), dim=0)

              if len(embeddings) > 0:
                  prototype = embeddings.mean(dim=0)
                  prototypes.append(prototype)

          if len(prototypes) > 0:
              prototypes = torch.stack(prototypes)
          else:
              raise Exception("Empty prototypes")

          if self.with_distractor:
            prototypes = torch.cat([prototypes, torch.zeros_like(prototypes)[0:1, :]], dim=0)
          support["prototypes"] = prototypes

          distances = torch.cdist(
              query["embeddings"].unsqueeze(0),
              prototypes.unsqueeze(0),
              p=2
          ).squeeze(0)

          distances = distances ** 2
          logits = -distances

          return logits


        elif query:
          support["embeddings"] = self.backbone(support["audio"])
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

          return logits


        elif positions and support_unlabeled:
          support["embeddings"] = self.backbone(support["audio"])
          support_unlabeled["embeddings"] = self.backbone(support_unlabeled["audio"])
          support["embeddings"].detach()
          support_unlabeled["embeddings"].detach()

          sup_embeddings = support["embeddings"].clone().requires_grad_(True)
          unlabeled_embeddings = support_unlabeled["embeddings"].clone().requires_grad_(True)

          support_embeddings = []
          for idx in range(len(support["classlist"])):
              embeddings = sup_embeddings[support["target"] == idx]
              support_embeddings.append(embeddings)
          support_embeddings = torch.stack(support_embeddings)

          prototypes = support_embeddings.mean(dim=1)
          if self.with_distractor:
            prototypes = torch.cat([prototypes, torch.zeros_like(prototypes)[0:1, :]], dim=0)
          support["prototypes"] = prototypes

          distances_list = []
          for idx, emb in enumerate(unlabeled_embeddings):
              selected_indices = positions[idx]
              new_prototypes = support["prototypes"][selected_indices]
              distances = torch.cdist(emb.unsqueeze(0), new_prototypes.unsqueeze(0), p=2).squeeze(0)
              distances_list.append(distances)

          distances = torch.stack(distances_list)
          distances = torch.squeeze(distances, dim=1)

          distances = distances ** 2
          logits = -distances

          return logits
        else:
          raise Exception("Wrong arguments!")
