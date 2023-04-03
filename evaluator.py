import torch
from torch.utils.data import DataLoader

class R2_SCORE:

    def __init__(self):
        pass

    def compute(self, y, y_pred):
        # Compute the target mean for each task
        y_mean = torch.mean(y, dim=0) #[1, task]

        # Compute the total sum of squares (TSS)
        tss = torch.sum(torch.pow(y - y_mean, 2), dim=0) #[1,task]

        # Compute the residual sum of squares (RSS)
        rss = torch.sum(torch.pow(y - y_pred, 2), dim=0) #[1,task]

        # Compute the R-squared
        r2 = 1 - (rss / tss) #[1,task]

        return r2