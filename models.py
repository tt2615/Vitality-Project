import torch
from torch import nn
from torch.utils.data import DataLoader

from evaluator import R2_SCORE, ADJUST_R2

class LR(nn.Module):

    def __init__(self, n_features, n_tasks):
        super(LR, self).__init__()

        # define parameters
        self.linear = nn.Linear(n_features, n_tasks, bias=True)

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
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(3).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y.to(device)
                pred = self.forward(x)
                print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y)
                
                n, p = x.shape[0], x.shape[1]
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred, n, p) #[1, task]

            return eval_loss, metrics_vals
        

class LLR(nn.Module):
    """lasso linear regression"""

    def __init__(self, n_features, n_tasks, lambda1=0.1, lambda2=0.1):
        super(LLR, self).__init__()

        # define parameters
        self.theta = torch.nn.Parameter(torch.randn(n_features))
        self.gamma = torch.nn.Parameter(torch.randn(n_tasks, n_features))
        self.bias = torch.nn.Parameter(torch.randn(n_tasks))
        self.lambda1 = torch.tensor(lambda1)
        self.lambda2 = torch.tensor(lambda2)

        # define evaluator
        self.evaluators = [R2_SCORE(), ADJUST_R2()]

    def forward(self, x):
        # print(x.shape, self.theta.shape, self.gamma.shape)
        weights = self.theta*self.gamma
        out = torch.matmul(x, torch.transpose(weights,0,1)) + self.bias
        return out
    
    def compute_loss(self, y_pred, y):
        l2_norm = torch.norm(y - y_pred) #l2 norm
        l2_norm_square = torch.pow(l2_norm, 2) #sum of l2 norm
        theta_reg = torch.sum(self.theta)
        gamma_reg = torch.sum(self.gamma)
        loss = l2_norm_square + self.lambda1*theta_reg + self.lambda2*gamma_reg
        return loss
    
    def eval(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(3).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y.to(device)
                pred = self.forward(x)
                print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y)
                
                n, p = x.shape[0], x.shape[1]
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred, n, p) #[1, task]

            return eval_loss, metrics_vals
