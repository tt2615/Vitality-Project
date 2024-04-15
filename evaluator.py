import torch

class R2_SCORE:
    def __call__(self, y, y_pred, *args):
        # Compute the target mean for each task
        y_mean = torch.mean(y, dim=0) #[1, task]

        # Compute the total sum of squares (TSS)
        tss = torch.sum(torch.pow(y - y_mean, 2), dim=0) #[1,task]

        # Compute the residual sum of squares (RSS)
        rss = torch.sum(torch.pow(y - y_pred, 2), dim=0) #[1,task]

        # Compute the R-squared
        r2 = 1 - (rss / tss) #[1,task]

        return r2
    
class ADJUST_R2:
    def __call__(self, y, y_pred, *args):
        n, p = args[0], args[1]
        # Compute the target mean for each task
        y_mean = torch.mean(y, dim=0) #[1, task]

        # Compute the total sum of squares (TSS)
        tss = torch.sum(torch.pow(y - y_mean, 2), dim=0) #[1,task]

        # Compute the residual sum of squares (RSS)
        rss = torch.sum(torch.pow(y - y_pred, 2), dim=0) #[1,task]

        # Compute the R-squared
        r2 = 1 - (rss / tss) #[1,task]

        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

        return adj_r2
    
class ACCURACY:
    def __call__(self, y, y_pred, *args):
        print(y)
        print(y_pred.max(1).indices)
        print(sum(y==y_pred))
        return 0
    

    