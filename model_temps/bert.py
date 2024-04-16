import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer

from evaluator import R2_SCORE, ADJUST_R2, ACCURACY, RECALL, PRECISION, F1

import pandas as pd

class Bert(nn.Module):
    def __init__(self, dim, cat_unique_count, embed_cols_count, num_cols_count, device):
        super(Bert, self).__init__()
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
        self.title_bert = BertModel.from_pretrained('bert-base-chinese')
        self.bert_linear = nn.Linear(768, dim, bias=True)
        

        ## cat input embedding module #'stock_code', 'item_author', 'article_author', 'article_source'\
        self.embedding_layer = []
        for i in range(embed_cols_count):
            self.embedding_layer.append(nn.Embedding(cat_unique_count[i], dim))

        ## num input network module #'item_views', 'item_comment_counts', 'article_likes', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism'
        self.num_feature_network = nn.Linear(num_cols_count, dim, bias=True)

        self.loss_fn = nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY()] #,RECALL(),PRECISION(),F1()

    def forward(self, text_input, non_text_input):
        #text representation
        title_output = self.title_bert(text_input[:,0,:], attention_mask=text_input[:,1,:]) #batch*768
        text_rep = self.bert_linear(title_output.pooler_output) #batch*dim

        title_att_score = torch.sum(title_output.attentions[-1],dim=1) #batch*len*len
        title_att_score = torch.sum(title_att_score,dim=1) #batch*len
        
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
        cat_reps = torch.zeros((non_text_input.shape[0],self.dim)).to(self.device)
        for i in range(self.embed_cols_count):
            single_embed_rep = self.embedding_layer[i](non_text_input[:,self.num_cols_count+i])
            cat_reps += single_embed_rep
        cat_rep = cat_reps/self.embed_cols_count #batch*dim

        final_rep = torch.cat((text_rep, num_rep, cat_rep), dim=1) #batch*3dim

        logits = self.classifier(final_rep) #batch*2
        return logits, title_att_score
    
    def compute_loss(self, y_pred, y):
        # l2_norm = torch.norm(y - y_pred) #l2 norm
        # loss = torch.pow(l2_norm, 2) #sum of l2 norm
        # return loss
        loss = self.loss_fn(y_pred, y)
        return loss
    
    def eval(self, eval_data:DataLoader, device, explain=True):
        with torch.no_grad():
            eval_loss = 0
            metrics_vals = {type(k).__name__:torch.zeros(1).to(device) for k in self.evaluators}

            preds, ys = torch.tensor([]), torch.tensor([])
            text, title_att_scores = [], torch.tensor([])
            for text_input, non_text_input, y in eval_data:
                y = y.squeeze().to(torch.long)
                pred, title_att_score = self.forward(text_input, non_text_input)
                eval_loss = self.compute_loss(pred, y)
                ys = torch.cat((ys,y))
                preds = torch.cat((preds, pred.max(1).indices))
                # print(ys, preds)

                if explain: #record attention scores for analysis
                    y_pos_index = (y==1).nonzero().squeeze()
                    
                    pos_title_att_score = title_att_score[y_pos_index]
                    # print(pos_title_att_score)
                    title_att_scores = torch.cat((title_att_scores, pos_title_att_score), axis=0).cpu().detach().numpy()
                    # print(title_att_scores)

                    pos_text_token = text_input[:,0,:].cpu().detach().numpy()
                    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                    for token in pos_text_token:
                        text.append(tokenizer.decode(token))
                
            for e in self.evaluators:
                metrics_vals[type(e).__name__] += e(ys, preds) #[1, task]

            #generate analysis report
            if explain and len(text)>0:
                report = pd.DataFrame({
                    'text': text,
                    'pred': preds,
                    'title_attention': list(title_att_scores),
                })
            else:
                report = None

            return eval_loss, metrics_vals, report
