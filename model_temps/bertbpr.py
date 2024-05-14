import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from evaluator import ACCURACY, CLASSIFICATION

# import numpy as np
import pandas as pd
from tqdm import tqdm

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
    

class BertAttBpr(nn.Module):
    def __init__(self, dim, cat_unique_count, user_unique_count, cat_cols_count, user_cols_count, num_cols_count, topic_num, device, bert='bert-base-chinese'):
        super(BertAttBpr, self).__init__()
        # define parameters
        self.dim = dim
        self.cat_cols_count = cat_cols_count
        self.num_cols_count = num_cols_count
        self.user_cols_count = user_cols_count
        self.topic_num = topic_num
        self.device = device
        self.bert = bert

        ## text input module
        # configuration = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=True)
        # configuration.hidden_dropout_prob = 0.8
        # configuration.attention_probs_dropout_prob = 0.8
        # self.title_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=configuration)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert)
        self.title_bert = BertModel.from_pretrained(bert, output_attentions=True)
        self.bert_linear = nn.Sequential(
            nn.Linear(768, dim*2, bias=True),
            nn.ReLU(),
            nn.Linear(dim*2, dim, bias=True),
            nn.Dropout(0.1) 
        )

        ## cat input embedding module #'item_author', 'article_author', 'article_source'
        self.post_embedding_layer = nn.ModuleList()
        for i in range(cat_cols_count):
            self.post_embedding_layer.append(nn.Embedding(cat_unique_count[i], dim))

        ## num input network module #'item_views', 'item_comment_counts', 'article_likes'
        self.network_layer = nn.ModuleList()
        for i in range(num_cols_count):
            self.network_layer.append(nn.Sequential(
            nn.Linear(1, dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(dim//2, dim, bias=True),
            nn.Dropout(0.1) 
        ))

        self.topic_layer = nn.Sequential(
            nn.Linear(topic_num, dim//2, bias=True),
            nn.ReLU(),
            nn.Linear(dim//2, dim, bias=True),
            nn.Dropout(0.1) 
        )

        self.user_attention_module = Attention(dim)
        self.post_attention_module = Attention(dim)
        # self.user_post_attention_module = Attention(dim)
        self.task_embedding = nn.Parameter(torch.rand(1,1,dim), requires_grad=True)

        ## user input embedding module # 'item_author', 'article_author', 'article_source'
        self.user_embedding_layer = nn.ModuleList()
        for i in range(user_cols_count):
            self.user_embedding_layer.append(nn.Embedding(user_unique_count[i], dim))

        # define evaluator
        self.evaluators = [ACCURACY(), CLASSIFICATION()]

    def forward(self, text_input, non_text_input, user_input):
        ## news representation

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
            stock_code_index                         52
            sentiment_score                    0.200904
            month_index                               5
            year_index                                4
            eastmoney_robo_journalism_index           1
            media_robo_journalism_index               1
            SMA_robo_journalism_index                 1
            topics_val1                        0.012618
            topics_val2                        0.012623
            topics_val3                        0.012636
            topics_val4                        0.214114
            topics_val5                        0.225046
            topics_val6                        0.012619
            topics_val7                        0.122692
            topics_val8                        0.387651    
        """

        # #num representation
        if self.num_cols_count>0:
            num_reps = torch.zeros((non_text_input.shape[0],1,self.dim)).to(self.device)
            for i in range(self.num_cols_count):
                # self.network_layer[i].to(self.device)
                num_rep = self.network_layer[i](non_text_input[:,i].unsqueeze(1)) #batch*dim  
                num_reps = torch.cat((num_reps, num_rep.unsqueeze(1)),dim=1)
            num_reps = num_reps[:,1:,:] #batch*1*dim
        else:
            num_reps = None

        #cat representation
        if self.cat_cols_count>0:
            cat_reps = torch.zeros((non_text_input.shape[0],1,self.dim)).to(self.device)
            for i in range(self.cat_cols_count):
                embed_rep = self.post_embedding_layer[i](non_text_input[:,self.num_cols_count+i].to(torch.int)) #batch*dim
                cat_reps = torch.cat((cat_reps, embed_rep.unsqueeze(1)),dim=1)
            cat_reps = cat_reps[:,1:,:] #batch*6*dim
            # print(cat_reps.shape)
        else:
            cat_reps = None

        #topic representation
        if self.topic_num>0:
            topic_rep = self.topic_layer(non_text_input[:,-self.topic_num:]).unsqueeze(1)
            # print(topic_rep.shape) #batch*1*dim
        else:
            topic_rep = None

        non_none_tensors = [] # Check if each tensor is not None, and if so, add it to the list
        if text_rep is not None:
            non_none_tensors.append(text_rep)
        if num_reps is not None:
            non_none_tensors.append(num_reps)
        if cat_reps is not None:
            non_none_tensors.append(cat_reps)
        if topic_rep is not None:
            non_none_tensors.append(topic_rep)
        post_reps = torch.cat(non_none_tensors, dim=1) #batch*12*dim
        # print(final_rep.shape)

        # attentioned_rep, feature_att_score = self.attention_module(final_rep, text_rep) #batch*1*dim
        post_attentioned_rep, post_feature_att_score = self.post_attention_module(post_reps, self.task_embedding.expand(post_reps.shape[0], -1, -1))
        # print(self.task_embedding)
        # print(attentioned_rep.shape)

        ## user representation
        """
            item_author_cate_index                    2
            article_author_index                      1
            article_source_cate_index                 1
        """
        user_reps = torch.zeros((user_input.shape[0],1,self.dim)).to(self.device)
        for i in range(self.user_cols_count):
            embed_rep = self.user_embedding_layer[i](user_input[:,i].to(torch.int)) #batch*dim
            user_reps = torch.cat((user_reps, embed_rep.unsqueeze(1)),dim=1)
        user_reps = user_reps[:,1:,:] #batch*9*dim
        # print(user_reps.shape)

        user_attentioned_rep, user_feature_att_score = self.user_attention_module(user_reps, self.task_embedding.expand(user_reps.shape[0], -1, -1))

        scores = torch.sigmoid(torch.bmm(user_attentioned_rep, post_attentioned_rep.transpose(1, 2)).squeeze())

        feature_att_score = torch.cat((post_feature_att_score, user_feature_att_score), dim=1)
        # print(feature_att_score.shape)

        return scores, feature_att_score, title_att_score
        # pos_score, p_feature_att_score, p_title_att_score = self.compute_score(pos_input)
        # neg_score, n_feature_att_score, n_title_att_score = self.compute_score(neg_input)

        # return pos_score, p_feature_att_score, p_title_att_score, neg_score, n_feature_att_score, n_title_att_score
    
    def train(self, data):
        pos_data, neg_data = data

        #---for pos data
        pos_text_input, pos_non_text_input, pos_user_input = pos_data

        #load data to device
        pos_text_input = pos_text_input.to(self.device)
        pos_non_text_input = pos_non_text_input.to(self.device)
        pos_user_input = pos_user_input.to(self.device)

        pos_scores, _, _ = self.forward(pos_text_input, pos_non_text_input, pos_user_input)

        #---for neg data
        neg_text_input, neg_non_text_input, neg_user_input = neg_data

        #load data to device
        neg_text_input = neg_text_input.to(self.device)
        neg_non_text_input = neg_non_text_input.to(self.device)
        neg_user_input = neg_user_input.to(self.device)

        neg_scores, _, _ = self.forward(neg_text_input, neg_non_text_input, neg_user_input)

        batch_loss = self.compute_loss(pos_scores, neg_scores)
            
        return batch_loss
    
    def compute_loss(self, pos_scores, neg_scores): ##BPR loss

        score_diff = pos_scores - neg_scores
        # print(score_diff)

        # Compute the BPR loss
        # print(pos_scores, neg_scores)
        print(score_diff)
        loss = -torch.log(torch.sigmoid(score_diff)).sum()

        # Add L2 regularization
        lambda_reg = 0
        reg_loss = lambda_reg * (torch.norm(pos_scores) + torch.norm(neg_scores))

        return loss + reg_loss


    def eval(self, eval_dataset, device, explain=False):
        valid_data, test_data = eval_dataset

        with torch.no_grad():
            ## compute validation loss
            eval_loss = 0
            valid_data = tqdm(valid_data, leave=False)
            valid_data.set_description("Evaluating model loss on validation set")
            for _, (pos_data, neg_data) in enumerate(valid_data):

                #---for pos data
                pos_text_input, pos_non_text_input, pos_user_input = pos_data

                #load data to device
                pos_text_input = pos_text_input.to(self.device)
                pos_non_text_input = pos_non_text_input.to(self.device)
                pos_user_input = pos_user_input.to(self.device)

                pos_scores, _, _ = self.forward(pos_text_input, pos_non_text_input, pos_user_input)

                #---for neg data
                neg_text_input, neg_non_text_input, neg_user_input = neg_data

                #load data to device
                neg_text_input = neg_text_input.to(self.device)
                neg_non_text_input = neg_non_text_input.to(self.device)
                neg_user_input = neg_user_input.to(self.device)

                neg_scores, _, _ = self.forward(neg_text_input, neg_non_text_input, neg_user_input)

                eval_loss += self.compute_loss(pos_scores, neg_scores)
            

            ## compute test metrics
            metrics_vals = {}
            total_scores, ys = torch.tensor([]), torch.tensor([])
            total_feature_att_scores, total_title_att_scores, total_text_input = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)
            text= []
            test_data = tqdm(test_data, leave=False)
            test_data.set_description("Testing model performance on test set")
            for _, data in enumerate(test_data):
                text_input, non_text_input, user_input, y = data
                text_input = text_input.to(self.device)
                non_text_input = non_text_input.to(self.device)
                user_input = user_input.to(self.device)

                ys = torch.cat((ys,y.cpu().detach()))

                scores, feature_att_score, title_att_score = self.forward(text_input, non_text_input, user_input)
                # Calculate the number of elements to set to 1
                scores = scores.cpu().detach()

                total_scores = torch.cat((total_scores, scores))
                total_feature_att_scores = torch.cat((total_feature_att_scores, feature_att_score))
                total_title_att_scores = torch.cat((total_title_att_scores, title_att_score))
                total_text_input = torch.cat((total_text_input, text_input))

            ## label data according to score
            x_percent = 0.01
            num_ones = int(ys.shape[0] * x_percent)
            # Sort the tensor in descending order
            sorted_pred, _ = torch.sort(total_scores, descending=True)
            # Threshold the tensor
            preds = torch.where(total_scores >= sorted_pred[num_ones], torch.tensor(1.0), torch.tensor(0.0))
            print(f'total pred 1s: {preds.sum()}')
                
            if explain: #record attention scores for analysis
                y_pos_index = (ys==1).nonzero()
                if y_pos_index.nelement()==1:
                    y_pos_index = y_pos_index.unsqueeze(0)

                pos_preds = preds[y_pos_index]
                # print(pos_preds)

                pos_feature_att_scores = total_feature_att_scores[y_pos_index]
                
                pos_title_att_scores = total_title_att_scores[y_pos_index]

                pos_text_token = total_text_input[y_pos_index,0,:]
                tokenizer = BertTokenizer.from_pretrained(self.bert)
                for sentence in pos_text_token:
                    for token in sentence:
                        text.append(tokenizer.decode(token))

            for e in self.evaluators:
                metrics_vals[type(e).__name__] = e(ys, preds) #[1, task]

            #generate analysis report
            if explain and len(text)>0:
                feature_list = ['text', 'sentiment', 'stock_code', 'month', 'year', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism', 'topic', 'item_author', 'article_author', 'article_source']
                report = pd.DataFrame({
                    'text': text,
                    'pred': list(pos_preds),
                    'title_attention': list(pos_title_att_scores.cpu().detach().numpy()),
                    'features': [feature_list]*(y_pos_index.shape[0]),
                    'feature_attention': list(pos_feature_att_scores.cpu().detach().numpy())
                })
            else:
                print('no positive data, no report generated')
                report = None
            
        return eval_loss, metrics_vals, report