import torch
import numpy as np

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
    
    def __repr__(self) -> str:
        return "ACCURACY"
    
class CLASSIFICATION:
    def __call__(self, y, y_pred, *args):
        y, y_pred = y.cpu(), y_pred.cpu()
        c_report = classification_report(y,y_pred)
        return c_report

    def __repr__(self) -> str:
        return "CLASSIFICATION"

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
    def __init__(self, k=None):
        self.k = k

    def __call__(self, y, y_pred, *args):
        datalen = args[0]
        y, y_pred = y.cpu(), y_pred.cpu()
        y = np.ravel(y)
        y_pred = np.ravel(y_pred)
         # Sort indices of y_pred in descending order of predicted scores
        idx_sorted_pred = np.argsort(y_pred)[::-1]
        # Sort indices of y in descending order of true scores
        idx_sorted_true = np.argsort(y)[::-1]
        
        # Calculate DCG for the predicted ordering
        def dcg(scores, idx_sorted, k):
            if not k:
                idx_sorted = idx_sorted
            elif k<1:
                idx_sorted = idx_sorted[:int(k*datalen)]
            else:
                idx_sorted = idx_sorted[:k]
            return np.sum((2**scores[idx_sorted] - 1) / np.log2(np.arange(1, len(idx_sorted) + 1) + 1))
        
        # Compute DCG for predicted and ideal (sorted by true relevance) orderings
        dcg_max = dcg(y, idx_sorted_true, self.k)
        dcg_pred = dcg(y, idx_sorted_pred, self.k)
        
        # Avoid division by zero if dcg_max is zero
        return dcg_pred / dcg_max if dcg_max > 0 else 0.0
    
    def __repr__(self) -> str:
        if self.k:
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
