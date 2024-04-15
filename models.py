#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig

from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from evaluator import R2_SCORE, ADJUST_R2, ACCURACY, RECALL, PRECISION, F1

class LR(nn.Module):

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
    
    def compute_loss(self, y_pred, y):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device):
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
        

class Bert(nn.Module):
    def __init__(self, dim, cat_unique_count, embed_cols_count, num_cols_count):
        super(Bert, self).__init__()
        # define parameters
        self.dim = dim
        self.embed_cols_count = embed_cols_count
        self.num_cols_count = num_cols_count

        ## text input module
        # configuration = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=True)
        # configuration.hidden_dropout_prob = 0.8
        # configuration.attention_probs_dropout_prob = 0.8
        # self.title_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=configuration)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.title_bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_linear = nn.Linear(768, dim, bias=True)
        

        ## cat input embedding module #'stock_code', 'item_author', 'article_author', 'article_source'\
        self.embedding_layer = []
        for i in range(embed_cols_count):
            self.embedding_layer.append(nn.Embedding(cat_unique_count[i], dim))

        ## num input network module #'item_views', 'item_comment_counts', 'article_likes', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism'
        self.num_feature_network = nn.Linear(num_cols_count, dim, bias=True)

        self.classifier = nn.Linear(3*dim, 2)

        # define loss
        self.loss_fn = nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY()] #,RECALL(),PRECISION(),F1()

    def forward(self, text_input, non_text_input):
        #text representation
        title_output = self.title_bert(text_input[:,0,:], attention_mask=text_input[:,1,:]) #batch*768
        text_rep = self.bert_linear(title_output[1]) #batch*dim

        
        """
        Non-text-input:
        item_views                   num
        item_comment_counts          num
        article_likes                num
        eastmoney_robo_journalism    cat
        media_robo_journalism        cat
        SMA_robo_journalism          cat
        stock_code_index             cat
        item_author_index            cat
        article_author_index         cat
        article_source_index         cat
        """

        #num representation
        num_rep = self.num_feature_network(non_text_input[:,:self.num_cols_count].to(torch.float32)) #batch*dim
        # print(num_rep.shape)

        #cat representation
        cat_reps = torch.zeros((non_text_input.shape[0],self.dim))
        for i in range(self.embed_cols_count):
            single_embed_rep = self.embedding_layer[i](non_text_input[:,self.num_cols_count+i])
            cat_reps += single_embed_rep
        cat_rep = cat_reps/self.embed_cols_count #batch*dim

        final_rep = torch.cat((text_rep, num_rep, cat_rep), dim=1) #batch*3dim

        logits = self.classifier(final_rep) #batch*2
        return logits
    
    def compute_loss(self, y_pred, y):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}


            preds, ys = torch.tensor([]), torch.tensor([])
            for text_input, non_text_input, y in eval_data:
                y = y.squeeze().to(torch.long)
                pred = self.forward(text_input, non_text_input)
                eval_loss = self.compute_loss(pred, y)
                ys = torch.cat((ys,y))
                preds = torch.cat((preds, pred.max(1).indices))
                # print(ys, preds)
                
            for e in self.evaluators:
                metrics_vals[type(e).__name__] += e(ys, preds) #[1, task]

            return eval_loss, metrics_vals


            acc=0
            for text_input, non_text_input, y in eval_data:
                pred_logit = self.forward(text_input, non_text_input)
                pred = pred_logit.max(1).indices
                ground_truth = y.squeeze().to(torch.long)
                eval_loss += self.compute_loss(pred_logit, ground_truth)
                acc = accuracy_score(pred, ground_truth)
                # print(classification_report(pred, ground_truth))

            return eval_loss, metrics_vals