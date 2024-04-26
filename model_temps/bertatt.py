import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer

from evaluator import R2_SCORE, ADJUST_R2, ACCURACY, RECALL, PRECISION, F1

import numpy as np
import pandas as pd

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x, q):
        queries = self.query(q) #batch*1*dim
        keys = self.key(x) #batch*feature*dim
        values = self.value(x) #batch*feature*dim
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5) #batch*1*feature
        attention = self.softmax(scores) #batch*1*feature
        weighted = torch.bmm(attention, values) #batch*1*dim
        return weighted, attention.squeeze(1)
    

class BertAtt(nn.Module):
    def __init__(self, dim, cat_unique_count, embed_cols_count, num_cols_count, device, bert='bert-base-chinese'):
        super(BertAtt, self).__init__()
        # define parameters
        self.dim = dim
        self.embed_cols_count = embed_cols_count
        self.num_cols_count = num_cols_count
        self.device = device

        ## text input module
        # configuration = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=True)
        # configuration.hidden_dropout_prob = 0.8
        # configuration.attention_probs_dropout_prob = 0.8
        # self.title_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=configuration)
        # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.title_bert = BertModel.from_pretrained(bert, 
                                                    output_attentions=True)
        self.bert_linear = nn.Linear(768, dim, bias=True)
        

        ## cat input embedding module #'stock_code', 'item_author', 'article_author', 'article_source', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism'\
        self.embedding_layer = nn.ModuleList()
        for i in range(embed_cols_count):
            self.embedding_layer.append(nn.Embedding(cat_unique_count[i], dim))

        ## num input network module #'item_views', 'item_comment_counts', 'article_likes',
        self.network_layer = nn.ModuleList()
        for i in range(num_cols_count):
            self.network_layer.append(nn.Linear(1, dim, bias=True))

        self.attention_module = Attention(dim)

        self.classifier = nn.Linear(dim, 2)

        # define loss
        self.loss_fn = nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY()] #RECALL(),PRECISION(),F1()

    def forward(self, text_input, non_text_input):

        #text representation
        title_output = self.title_bert(text_input[:,0,:], attention_mask=text_input[:,1,:]) #batch*768
        text_rep = title_output.pooler_output #batch*768
        # text_rep = torch.flatten(text_rep, start_dim=1) #batch*(len*768)
        text_rep = self.bert_linear(text_rep).unsqueeze(1) #batch*1*dim
        # print(text_rep.shape)

        #extract attention: title_output.attentions has 12 (layers) of (batch*head(12)*len*len)
        title_att_score = torch.sum(title_output.attentions[-1],dim=1) #batch*len*len
        title_att_score = torch.sum(title_att_score,dim=1) #batch*len
        # print(title_att_score.shape)

        """
        Non-text-input:
            stock_code_index                   
            item_author_index                  
            article_author_index               
            article_source_index               
            eastmoney_robo_journalism_index    
            media_robo_journalism_index        
            SMA_robo_journalism_index          
            month_index                        
            year_index                         
        """

        #num representation
        num_reps = torch.zeros((non_text_input.shape[0],1,self.dim)).to(self.device)
        for i in range(self.num_cols_count):
            # self.network_layer[i].to(self.device)
            num_rep = self.network_layer[i](non_text_input[:,i].unsqueeze(1).to(torch.float)) #batch*dim  
            num_reps = torch.cat((num_reps, num_rep.unsqueeze(1)),dim=1)
        num_reps = num_reps[:,1:,:] #batch*3*dim
        # print(num_reps.shape)

        #cat representation
        cat_reps = torch.zeros((non_text_input.shape[0],1,self.dim)).to(self.device)
        for i in range(self.embed_cols_count):
            embed_rep = self.embedding_layer[i](non_text_input[:,self.num_cols_count+i].to(torch.int)) #batch*dim
            cat_reps = torch.cat((cat_reps, embed_rep.unsqueeze(1)),dim=1)
        cat_reps = cat_reps[:,1:,:] #batch*7*dim
        # print(cat_reps.shape)

        final_rep = torch.cat((text_rep, num_reps, cat_reps), dim=1) #batch*11*dim
        # print(final_rep.shape)

        attentioned_rep, feature_att_score = self.attention_module(final_rep, text_rep) #batch*1*dim
        # print(attentioned_rep.shape)

        logits = self.classifier(attentioned_rep.squeeze()) #batch*2
        # print(logits.shape)

        return logits, feature_att_score, title_att_score
    
    def compute_loss(self, y_pred, y):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device, explain=False):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}

            preds, ys = torch.tensor([]), torch.tensor([])
            text, pos_feature_att_scores, pos_title_att_scores = [], torch.tensor([]).to(device), torch.tensor([]).to(device)
            for text_input, non_text_input, y in eval_data:
                
                text_input = text_input.to(device)
                non_text_input = non_text_input.to(device)
                y = y.squeeze().to(torch.long).to(device)

                pred, feature_att_score, title_att_score = self.forward(text_input, non_text_input)

                eval_loss = self.compute_loss(pred, y)

                ys = torch.cat((ys,y.cpu().detach()))
                preds = torch.cat((preds, pred.cpu().detach().max(1).indices))
                # print(ys, preds)
                
                if explain: #record attention scores for analysis
                    y_pos_index = (y==0).nonzero().squeeze()
                    if len(y_pos_index)>0:
                        pos_feature_att_score = feature_att_score[y_pos_index]
                        # print(pos_feature_att_score)
                        pos_feature_att_scores = torch.cat((pos_feature_att_scores, pos_feature_att_score))
                        # print(feature_att_scores)
                        
                        pos_title_att_score = title_att_score[y_pos_index]
                        # print(pos_title_att_score)
                        pos_title_att_scores = torch.cat((pos_title_att_scores, pos_title_att_score))
                        # print(title_att_scores)

                        pos_text_token = text_input[y_pos_index,0,:]
                        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                        for token in pos_text_token:
                            text.append(tokenizer.decode(token))

            for e in self.evaluators:
                metrics_vals[type(e).__name__] += e(ys, preds) #[1, task]

            #generate analysis report
            if explain and len(text)>0:
                feature_list = ['text', 'item_views', 'item_comment_counts', 'article_likes', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism', 'stock_code_index', 'item_author_index', 'article_author_index', 'article_source_index']
                report = pd.DataFrame({
                    'text': text,
                    'pred': preds,
                    'title_attention': list(pos_title_att_scores.cpu().detach().numpy()),
                    'features': [feature_list]*len(y_pos_index),
                    'feature_attention': list(pos_feature_att_scores.cpu().detach().numpy())
                })
            else:
                report = None

            return eval_loss, metrics_vals, report