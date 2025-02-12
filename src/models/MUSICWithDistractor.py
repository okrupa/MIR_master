import torch

from models.MUSIC import MUSIC

class MUSICWithDistractor(MUSIC):

    def step(self, batch, batch_idx, tag: str):
        torch.set_grad_enabled(True)
        if self.with_distractor:
            support, unlabeled, query, non_distractor = batch
        else:
            support, unlabeled, query = batch

        num_unlabeled = len(unlabeled["audio"])
        num_class = len(support['classlist'])
        ori_index = [x for x in range(num_unlabeled)]
        _POSITION = [[] for _ in range(num_unlabeled)]
        POSITION = [list(range(num_class+1)) for _ in range(num_unlabeled)]
        for n_c in range(num_class):
          unlabel_logits = self.protonet(support, support_unlabeled = unlabeled, positions = POSITION)
          nl_pred, unselect_idx = self.protonet.get_preds_position(unlabel_logits, POSITION, _POSITION, thres=1)
          select_idx = [x for x in ori_index if x not in unselect_idx]
          unlabel_logits_selected = unlabel_logits[select_idx]
          nl_pred = nl_pred[select_idx]
          POSITION = [POSITION[i] for i in select_idx]
          _POSITION = [_POSITION[i] for i in select_idx]

          nl_pred = nl_pred.to(self.device)

          loss_NG = self.neg_loss(unlabel_logits_selected, nl_pred)
          loss_MINI = self.mini_entropy_loss(unlabel_logits_selected)

          loss = self.neg_loss(unlabel_logits_selected, nl_pred) + self.mini_entropy_loss(unlabel_logits_selected)

          loss.backward()
          self.optimizer_NL.step()
          self.optimizer_NL.zero_grad()


        positions_tensor = torch.tensor(POSITION)

        unique_values = torch.unique(positions_tensor)

        index_mapping = {val.item(): idx for idx, val in enumerate(unique_values)}

        unlabeled_target = torch.tensor([index_mapping[val.item()] for val in positions_tensor])

        unlabeled["target"] = unlabeled_target
        logits = self.protonet(support, query = query, support_unlabeled = unlabeled)

        loss = self.loss(logits, query["target"])

        output = {"loss": loss}
        for k, metric in self.metrics.items():
            output[k] = metric(logits, query["target"])

        for k, v in output.items():
            self.log(f"{k}/{tag}", v)
        return output


    def predict_with_unlabeled(self, support, unlabeled, query):
        torch.set_grad_enabled(True)
        self.optimizer_NL = torch.optim.Adam(self.parameters(), lr=1e-5)

        num_unlabeled = len(unlabeled["audio"])
        num_class = len(support['classlist'])
        ori_index = [x for x in range(num_unlabeled)]
        _POSITION = [[] for _ in range(num_unlabeled)]
        POSITION = [list(range(num_class+1)) for _ in range(num_unlabeled)]
        for n_c in range(num_class):
          unlabel_logits = self.protonet(support, support_unlabeled = unlabeled, positions = POSITION)
          nl_pred, unselect_idx = self.protonet.get_preds_position(unlabel_logits, POSITION, _POSITION, thres=1)
          select_idx = [x for x in ori_index if x not in unselect_idx]
          unlabel_logits_selected = unlabel_logits[select_idx]
          nl_pred = nl_pred[select_idx]
          POSITION = [POSITION[i] for i in select_idx]
          _POSITION = [_POSITION[i] for i in select_idx]

          nl_pred = nl_pred.to(self.device)

          loss_NG = self.neg_loss(unlabel_logits_selected, nl_pred)
          loss_MINI = self.mini_entropy_loss(unlabel_logits_selected)

          loss = self.neg_loss(unlabel_logits_selected, nl_pred) + self.mini_entropy_loss(unlabel_logits_selected)

          loss.backward()
          self.optimizer_NL.step()
          self.optimizer_NL.zero_grad()

        positions_tensor = torch.tensor(POSITION)

        unique_values = torch.unique(positions_tensor)

        index_mapping = {val.item(): idx for idx, val in enumerate(unique_values)}

        unlabeled_target = torch.tensor([index_mapping[val.item()] for val in positions_tensor])

        unlabeled["target"] = unlabeled_target
        logits = self.protonet(support, query = query, support_unlabeled = unlabeled)

        return logits
