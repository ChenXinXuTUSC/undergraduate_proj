import torch
import torch.nn as nn

class BalancedLoss(nn.Module):
    '''from pytorch document: 
    This loss combines a Sigmoid layer and the
    BCELoss in one single class. This  version
    is more numerically  stable  than using  a
    plain Sigmoid followed by a BCELoss as, by
    combining the operations into  one  layer,
    we take advantage of the log-sum-exp trick
    for numerical stability.
    '''

    NUM_LABELS = 2

    def __init__(self):
        super().__init__()
        self.crit = nn.BCEWithLogitsLoss()

    def forward(self, logits, label):
        assert torch.all(label < self.NUM_LABELS)
        loss = torch.scalar_tensor(0.).to(logits)
        for i in range(self.NUM_LABELS):
            target_mask = label == i
            if torch.any(target_mask):
                loss += self.crit(logits[target_mask], label[target_mask].to(
                    torch.float)) / self.NUM_LABELS
        return loss
