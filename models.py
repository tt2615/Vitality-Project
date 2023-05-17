import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertConfig, BertModel, BertTokenizer

from evaluator import R2_SCORE, ADJUST_R2, CONF_MATRIX, ACCURACY, RECALL, PRECISION, F1
class LR(nn.Module):
    "normal linear regression"

    def __init__(self, n_features, n_tasks):
        super(LR, self).__init__()
        self.n_features = n_features
        self.n_tasks = n_tasks

        # define parameters
        self.linear = nn.Linear(n_features, n_tasks, bias=True)

        self.loss_fn = nn.MSELoss(reduction='mean')

        # define evaluator
        self.evaluators = [R2_SCORE(), ADJUST_R2()]

    def forward(self, x):
        out = self.linear(x)
        return out
    
    def compute_loss(self, y_pred, y, *args):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def evaluate(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(self.n_tasks).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y.to(device)
                pred = self.forward(x)
                # print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y)
                
                n, p = x.shape[0], x.shape[1] #x: number of data, y: dimension of data
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred, n, p) #[1, task]

            return eval_loss, metrics_vals
        
class LogR(nn.Module):
    """logistic regression model"""

    def __init__(self, n_features, n_tasks):
        super(LogR, self).__init__()
        self.n_features = n_features
        self.n_tasks = n_tasks

        # define parameters
        self.linear = nn.Linear(n_features, n_tasks, bias=True)

        self.loss_fn = nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY(), RECALL(), PRECISION(), F1()]

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out
    
    def compute_loss(self, y_pred, y, *args):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def evaluate(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(self.n_tasks).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y.to(device)
                pred = self.forward(x)
                print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y)
                
                n, p = x.shape[0], x.shape[1]
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred, n, p) #[1, task]

                conf_matrix = CONF_MATRIX()
                print(conf_matrix(y.cpu(), pred.cpu()))

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
    
    def compute_loss(self, y_pred, y, *args):
        l2_norm = torch.norm(y - y_pred) #l2 norm
        l2_norm_square = torch.pow(l2_norm, 2) #sum of l2 norm
        theta_reg = torch.sum(self.theta)
        gamma_reg = torch.sum(self.gamma)
        loss = l2_norm_square + self.lambda1*theta_reg + self.lambda2*gamma_reg
        return loss
    
    def evaluate(self, eval_data:DataLoader, device):
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


class SLR(nn.Module):
    """Single task linear regression"""
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
    
    def compute_loss(self, y_pred, y, *args):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        y = y[:,self.y_index].unsqueeze(1)
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def evaluate(self, eval_data:DataLoader, device):
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

class DeepMTL(nn.Module):
    """Deep MTL model"""
    def __init__(self, num_author, num_company, num_sentiment, num_topic):
        super(DeepMTL, self).__init__()

        # define model
        configuration = BertConfig.from_pretrained('bert-base-chinese', num_labels=2, output_hidden_states=True, output_attentions=True)
        hidden_size = configuration.hidden_size

        self.language_encoder = BertModel.from_pretrained('bert-base-chinese', config=configuration)
        # self.content_encoder = BertModel.from_pretrained('bert-base-chinese', config=configuration)
        self.author_embedding = nn.Embedding(num_author, hidden_size)
        self.company_embedding = nn.Embedding(num_author, hidden_size)
        self.sentiment_embedding = nn.Embedding(num_author, hidden_size)
        self.topic_embedding = nn.Embedding(num_topic, hidden_size)

        self.t1_output = nn.Linear(hidden_size, 2)
        self.t2_output = nn.Linear(hidden_size, 2)
        self.t3_output = nn.Linear(hidden_size, 2)
        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY(), RECALL(), PRECISION(), F1()]

    def forward(self, x):
        # process language information
        title_embedding = self.language_encoder(x[0])
        content_embdding = self.language_encoder(x[1])

        # process categorical information
        author_embedding = self.author_embedding(x[2])
        company_embedding = self.company_embedding(x[3])
        sentiment_embedding = self.sentiment_embedding(x[4])
        topic_embedding = self.topic_embedding(x[5])
        categorical_embedding = author_embedding + company_embedding + sentiment_embedding + topic_embedding

        t1_input = title_embedding + categorical_embedding
        t2_input = title_embedding + content_embdding + categorical_embedding
        t3_input = title_embedding + content_embdding + categorical_embedding

        bert_out = self.topic_encoder(x)
        print(bert_out.shape)
        out1 = self.t1_output(t1_input).softmax()
        print(out1.shape)
        out2 = self.t2_output(t2_input).softmax()
        print(out2.shape)
        out3 = self.t3_output(t3_input).softmax()
        print(out3.shape)
        return out1, out2, out3
    
    def compute_loss(self, y_pred, y, *args):
        w1, w2, w3 = args[0], args[1], args[2]
        print(y_pred[:,0].shape, y[:,0].shape)
        loss1 = self.loss_fn(y_pred[:,0], y[:,0])
        loss2 = self.loss_fn(y_pred[:,1], y[:,1])
        loss3 = self.loss_fn(y_pred[:,2], y[:,2])
        return w1*loss1 + w2*loss2 + w3*loss3
    

    def evaluate(self, eval_data:DataLoader, device, *args):
        w1, w2, w3 = args[0], args[1], args[2]
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y[:,self.y_index].unsqueeze(1)
                y = y.to(device)
                pred = self.forward(x)
                # print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y, w1, w2, w3)
                
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred) #[1, task]

            return eval_loss, metrics_vals
        
class Deep(nn.Module):
    """Deep model for single task"""
    def __init__(self, num_author, num_company, num_sentiment, num_topic, hidden_size):
        super(Deep, self).__init__()

        # define model
        configuration = BertConfig.from_pretrained('bert-base-chinese')
        self.language_encoder = BertModel.from_pretrained('bert-base-chinese', config=configuration)
        # self.content_encoder = BertModel.from_pretrained('bert-base-chinese', config=configuration)
        self.dim_reducer = nn.Linear(configuration.hidden_size, hidden_size)
        
        self.author_embedding = nn.Embedding(num_author, hidden_size)
        self.company_embedding = nn.Embedding(num_company, hidden_size)
        self.sentiment_embedding = nn.Embedding(num_sentiment, hidden_size)
        self.topic_mlp = nn.Linear(num_topic, hidden_size)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1)
        ) 
        # define loss function
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # define evaluator
        # self.evaluators = [ACCURACY(), RECALL(), PRECISION(), F1()]

    def forward(self, x):
        # process language information
        _, title_embedding = self.language_encoder(x[:,0:32].int(), attention_mask=x[:,32:64].int(), return_dict=False) #batch * hd_size
        title_embedding = self.dim_reducer(title_embedding)
        #_, content_embdding = self.content_encoder(x[:,64:320].int(), attention_mask=x[:,320:576].int(), return_dict=False) #batch * hd_size

        # process categorical information
        author_embedding = self.author_embedding(x[:,576].int())
        company_embedding = self.company_embedding(x[:,577].int())
        sentiment_embedding = self.sentiment_embedding(x[:,578].int())
        topic_embedding = self.topic_mlp(x[:,579:584])
        categorical_embedding = author_embedding + company_embedding + sentiment_embedding + topic_embedding

        input = title_embedding + categorical_embedding #batch * hd_size + content_embdding

        out = self.output_layer(input) #batch * 2
        return out
    
    def compute_loss(self, y_pred, y):
        return self.loss_fn(y_pred[:,0], y[:,0])
    

    def evaluate(self, eval_data:DataLoader, device, *args):
        w1, w2, w3 = args[0], args[1], args[2]
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}

            for x, y in eval_data:
                x = x.to(device)
                y = y[:,self.y_index].unsqueeze(1)
                y = y.to(device)
                pred = self.forward(x)
                # print(x.shape, y.shape, pred.shape)

                eval_loss = self.compute_loss(pred, y, w1, w2, w3)
                
                for e in self.evaluators:
                    metrics_vals[type(e).__name__] += e(y, pred) #[1, task]

            return eval_loss, metrics_vals
    



        