import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluator import R2_SCORE, ADJUST_R2

class SLR(nn.Module):

    def __init__(self, n_features, y_index):
        super(SLR, self).__init__()
        self.n_features = n_features
        self.y_index = y_index

        # define parameters
        self.linear = nn.Linear(n_features, 1, bias=True)

        self.loss_fn = nn.MSELoss(reduction='mean')

        # define evaluator
        self.evaluators = [R2_SCORE(), ADJUST_R2()]

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def compute_loss(self, y_pred, y):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        y = y[:,self.y_index].unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y[:,self.y_index].unsqueeze(1)
                y = y.to(device)
                pred = self.forward(x)
                # print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y)
                
                n, p = x.shape[0], x.shape[1]
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred, n, p) #[1, task]

            return eval_loss, metrics_vals
    