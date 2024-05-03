import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertConfig, BertTokenizer

from evaluator import R2_SCORE, ADJUST_R2, ACCURACY, RECALL, PRECISION, F1

import pandas as pd

class Bert(nn.Module):
    def __init__(self, dim, cat_unique_count, embed_cols_count, num_cols_count, device, bert='bert-base-chinese'):
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
        self.title_bert = BertModel.from_pretrained(bert, output_attentions=True)
        self.bert_linear = nn.Linear(768, dim, bias=True)
        

        ## cat input embedding module #'stock_code', 'item_author', 'article_author', 'article_source'\
        self.embedding_layer = nn.ModuleList()
        for i in range(embed_cols_count):
            self.embedding_layer.append(nn.Embedding(cat_unique_count[i], dim))

        ## num input network module #'item_views', 'item_comment_counts', 'article_likes', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism'
        self.num_feature_network = nn.Linear(num_cols_count, dim, bias=True)

        self.classifier = nn.Linear(2*dim, 2)

        self.loss_fn = nn.CrossEntropyLoss()

        # define evaluator
        self.evaluators = [ACCURACY()] #,RECALL(),PRECISION(),F1()

    def forward(self, text_input, non_text_input):
        #text representation
        title_output = self.title_bert(text_input[:,0,:], attention_mask=text_input[:,1,:]) #batch*768
        text_rep = self.bert_linear(title_output.pooler_output) #batch*dim
        # print(title_output)

        title_att_score = torch.sum(title_output.attentions[-1],dim=1) #batch*len*len
        title_att_score = torch.sum(title_att_score,dim=1) #batch*len
        # title_att_score=0
        
        """
        Non-text-input:
            stock_code_index                    44
            item_author_index                   38
            article_author_index                 0
            article_source_index               327
            month_index                          8
            year_index                           2
            eastmoney_robo_journalism_index      1
            media_robo_journalism_index          1
            SMA_robo_journalism_index                 
        """

        #num representation
        # num_rep = self.num_feature_network(non_text_input[:,:self.num_cols_count].to(torch.float32)) #batch*dim
        # print(num_rep.shape)

        #cat representation
        cat_reps = torch.zeros((non_text_input.shape[0],self.dim)).to(self.device)
        for i in range(self.embed_cols_count):
            single_embed_rep = self.embedding_layer[i](non_text_input[:,self.num_cols_count+i].to(torch.int))
            cat_reps += single_embed_rep
        cat_rep = cat_reps/self.embed_cols_count #batch*dim

        # final_rep = torch.cat((text_rep, num_rep, cat_rep), dim=1) #batch*3dim
        final_rep = torch.cat((text_rep, cat_rep), dim=1)

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
            y_pos_len, pos_preds = 0, torch.tensor([])
            text, pos_title_att_scores = [], torch.tensor([]).to(device)
            for x, y in eval_data:

                text_input, non_text_input = x
                
                text_input = text_input.to(device)
                non_text_input = non_text_input.to(device)
                y = y.squeeze().to(torch.long).to(device)

                pred, title_att_score = self.forward(text_input, non_text_input)

                eval_loss = self.compute_loss(pred, y)

                ys = torch.cat((ys,y.cpu().detach()))
                preds = torch.cat((preds, pred.cpu().detach().max(1).indices))
                # print(ys, preds)
                
                if explain: #record attention scores for analysis
                    y_pos_index = (y==1).nonzero().squeeze().cpu().detach()
                    # print('---------')
                    # print(y_pos_index)
                    # print(len(y_pos_index.size()))
                    if y_pos_index.nelement() > 0:
                        y_pos_len += y_pos_index.nelement()
                        # print(y_pos_index)
                        # print(preds[y_pos_index])
                        if y_pos_index.nelement()==1:
                            y_pos_index = y_pos_index.unsqueeze(0)

                        pos_preds = torch.cat((pos_preds, preds[y_pos_index]))
                        # print(pos_preds)
                        
                        pos_title_att_score = title_att_score[y_pos_index]
                        # print(pos_title_att_score)
                        pos_title_att_scores = torch.cat((pos_title_att_scores, pos_title_att_score), 0)
                        # print(pos_title_att_scores)

                        pos_text_token = text_input[y_pos_index,0,:]
                        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
                        for token in pos_text_token:
                            text.append(tokenizer.decode(token))
                        # print(text)

            for e in self.evaluators:
                metrics_vals[type(e).__name__] += e(ys, preds) #[1, task]

            # print(y_pos_len)
            # print(len(pos_preds))
            # print(pos_title_att_scores.shape)
            # print(len(text))

            #generate analysis report
            if explain and len(text)>0:
                report = pd.DataFrame({
                    'text': text,
                    'pred': pos_preds,
                    'title_attention': list(pos_title_att_scores.cpu().detach().numpy()),
                })
            else:
                print('no positive data, no report generated')
                report = None

            return eval_loss, metrics_vals, report
