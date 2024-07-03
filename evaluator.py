import torch

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, classification_report, ndcg_score

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
    
class CONF_MATRIX:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        cm = confusion_matrix(y, y_pred)
        return cm
    
class ACCURACY:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        accuracy = accuracy_score(y, y_pred)
        return accuracy
    
class CLASSIFICATION:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        c_report = classification_report(y,y_pred)
        return c_report

class RECALL:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        recall = recall_score(y, y_pred)
        return recall
    
class PRECISION:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        precision = precision_score(y, y_pred)
        return precision
    
class F1:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        f1 = f1_score(y, y_pred)
        return f1


class NDCG:
    def __init__(self, k):
        self.k = k

    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu().unsqueeze(0), y_pred.cpu().unsqueeze(0)
        if self.k!=-1:
            ndcg = ndcg_score(y, y_pred, k=self.k)
        else:
            ndcg = ndcg_score(y, y_pred)
        return ndcg
    
    def __repr__(self) -> str:
        if self.k != -1:
            return f"NDCG@{self.k}"
        else:
            return f"NDCG"
    

# def ndcg_factory(k) :
#     class NDCGATK(NDCG): 
#         def __call__(self, y, y_pred, k, *args):
#             y, y_pred = y.cpu().unsqueeze(0), y_pred.cpu().unsqueeze(0)
#             ndcg = ndcg_score(y, y_pred, k)
#             return ndcg
        
#     NDCGATK.__name__ = f"NDCG@{k}"
#     return NDCGATK
