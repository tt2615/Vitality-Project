import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from evaluator import ACCURACY, CLASSIFICATION, NDCG

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
    

class IncBertAttBpr(nn.Module):
    def __init__(self, dim,
        post_ft_unique_count,
        author_ft_unique_count,
        post_ft_count,
        author_ft_count,
        device,
        drop_rate,
        bert='bert-base-chinese',
        bert_freeze = False):
        super(IncBertAttBpr, self).__init__()
        # define parameters
        self.dim = dim
        self.post_ft_count = post_ft_count
        self.author_ft_count = author_ft_count
        self.post_ft_unique_count = post_ft_unique_count
        self.author_ft_unique_count = author_ft_unique_count
        self.device = device
        self.bert = bert
        self.bert_freeze = bert_freeze
        self.drop_rate = drop_rate
        self.num_heads = 2

        ## text input module
        # configuration = BertConfig.from_pretrained('bert-base-chinese', output_hidden_states=True, output_attentions=True)
        # configuration.hidden_dropout_prob = 0.8
        # configuration.attention_probs_dropout_prob = 0.8
        # self.title_bert = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=configuration)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert)
        self.title_bert = BertModel.from_pretrained(bert, output_attentions=True)
        self.bert_linear = nn.Sequential(
            nn.Linear(768, dim*2, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(dim*2, dim, bias=True),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate) 
        )
        if self.bert_freeze:
            for param in self.title_bert.parameters():
                param.requires_grad = False

        ## 
        self.post_embedding_layer = nn.ModuleList()
        for i in range(post_ft_count):
            self.post_embedding_layer.append(nn.Embedding(post_ft_unique_count[i], dim))

        self.author_embedding_layer = nn.ModuleList()
        for i in range(author_ft_count):
            self.author_embedding_layer.append(nn.Embedding(author_ft_unique_count[i], dim))


        # self.author_attention_module = Attention(dim)
        # self.post_attention_module = Attention(dim)
        self.author_attention_module = nn.MultiheadAttention(dim, self.num_heads, dropout=self.drop_rate, batch_first=True)
        self.post_attention_module = nn.MultiheadAttention(dim, self.num_heads, dropout=self.drop_rate, batch_first=True)

        self.task_embedding = nn.Parameter(torch.rand(1,1,dim), requires_grad=True)

        self.post_dropout = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
        )
        self.author_dropout = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
        )

        # define evaluator
        self.evaluators = [ACCURACY(), CLASSIFICATION(), NDCG(10), NDCG(0.01), NDCG(0.05), NDCG()]

    def forward(self, text_input, post_input, author_input):
        ## post representation

        #text representation
        title_output = self.title_bert(text_input[:,0,:], attention_mask=text_input[:,1,:]) #batch*768
        text_rep = title_output.pooler_output #batch*768
        # text_rep = torch.flatten(text_rep, start_dim=1) #batch*(len*768)
        text_rep = self.bert_linear(text_rep).unsqueeze(1) #batch*1*dim
        # print(text_rep.shape)

        #extract attention
        attentions = title_output.attentions  # This is a tuple of attention matrices from each layer
 
        # Concatenate attentions from all layers (stack them)
        all_layer_attentions = torch.stack(attentions, dim=0)  # Shape: [num_layers, batch_size, num_heads, seq_len, seq_len]
        # print(all_layer_attentions.shape)

        # Average across attention heads (shape: [num_layers, batch_size, seq_len, seq_len])
        avg_attention_heads = all_layer_attentions.mean(dim=2)  # Shape: [num_layers, batch_size, seq_len, seq_len]

        # Average across all layers (shape: [batch_size, seq_len, seq_len])
        avg_attention_layers = avg_attention_heads.mean(dim=0)  # Shape: [batch_size, seq_len, seq_len]

        # Concentrated attention score for each token is the sum of attention values across all positions
        title_att_score = avg_attention_layers.sum(dim=1)  # Shape: [seq_len]

        """
        non_text post feature:
            'month', 
            'IndustryCode1',
            'IndustryCode2',
            'sentiment',
            'topic',
        """

        #post feature representation
        # Initialize an empty list to store embeddings
        non_text_reps = []
        post_input = post_input.long()
        # Iterate over the inputs and corresponding embedding layers
        for i in range(self.post_ft_count):
            # Apply the embedding layer to the input
            embed_rep = self.post_embedding_layer[i](post_input[:, i])  # shape: (batch_size, dim)
            non_text_reps.append(embed_rep)
        non_text_reps = torch.stack(non_text_reps, dim=1) #batch*5*dim

        post_reps = torch.cat([text_rep, non_text_reps], dim=1) #batch*6*dim

        # attentioned_rep, feature_att_score = self.attention_module(final_rep, text_rep) #batch*1*dim
        # post_attentioned_rep, post_feature_att_score = self.post_attention_module(post_reps, self.task_embedding.expand(post_reps.shape[0], -1, -1))
        post_attentioned_rep, post_feature_att_score = self.post_attention_module(post_reps, post_reps, post_reps)
        post_attentioned_rep, post_feature_att_score = post_attentioned_rep.mean(dim=1), post_feature_att_score.mean(dim=1)
        post_attentioned_rep = self.post_dropout(post_attentioned_rep.squeeze())
        # print(self.task_embedding)
        # print(attentioned_rep.shape)

        """
        author feature:
            'eastmoney_robo_journalism',
            'media_robo_journalism',
            'SMA_robo_journalism',
            'item_author_index_rank',
            'article_author_index_rank',
            'article_source_index_rank'
        """

        author_reps = []
        author_input = author_input.long()
        # Iterate over the inputs and corresponding embedding layers
        for i in range(self.author_ft_count):
            # Apply the embedding layer to the input
            embed_rep = self.author_embedding_layer[i](author_input[:, i])  # shape: (batch_size, dim)
            author_reps.append(embed_rep)
        author_reps = torch.stack(author_reps, dim=1) #batch*6*dim
        # print(author_reps.shape)

        # author_attentioned_rep, author_feature_att_score = self.author_attention_module(author_reps, self.task_embedding.expand(author_reps.shape[0], -1, -1))
        author_attentioned_rep, author_feature_att_score = self.author_attention_module(author_reps, author_reps, author_reps)
        author_attentioned_rep, author_feature_att_score = author_attentioned_rep.mean(dim=1), author_feature_att_score.mean(dim=1)
        author_attentioned_rep = self.author_dropout(author_attentioned_rep.squeeze())

        # scores = torch.sigmoid(torch.bmm(post_attentioned_rep, author_attentioned_rep.transpose(1, 2)).squeeze())
        scores = torch.sum(post_attentioned_rep * author_attentioned_rep, dim=1)

        feature_att_score = torch.cat((post_feature_att_score, author_feature_att_score), dim=1)
        # print(feature_att_score.shape)

        return scores, feature_att_score, title_att_score, post_attentioned_rep, author_attentioned_rep
        # pos_score, p_feature_att_score, p_title_att_score = self.compute_score(pos_input)
        # neg_score, n_feature_att_score, n_title_att_score = self.compute_score(neg_input)

        # return pos_score, p_feature_att_score, p_title_att_score, neg_score, n_feature_att_score, n_title_att_score
    
    def train(self, data):
        pos_data, neg_data = data

        #---for pos data
        pos_text_input, pos_non_text_input, pos_author_input = pos_data

        #load data to device
        pos_text_input = pos_text_input.to(self.device)
        pos_non_text_input = pos_non_text_input.to(self.device)
        pos_author_input = pos_author_input.to(self.device)

        pos_scores, _, _, pos_post_embed, pos_author_embed = self.forward(pos_text_input, pos_non_text_input, pos_author_input)

        #---for neg data
        neg_text_input, neg_non_text_input, neg_author_input = neg_data

        #load data to device
        neg_text_input = neg_text_input.to(self.device)
        neg_non_text_input = neg_non_text_input.to(self.device)
        neg_author_input = neg_author_input.to(self.device)

        neg_scores, _, _, neg_post_embed, neg_author_embed = self.forward(neg_text_input, neg_non_text_input, neg_author_input)

        # batch_loss = self.compute_loss(pos_scores, neg_scores, (pos_post_embed, pos_author_embed, neg_post_embed, neg_author_embed))
        batch_loss = self.compute_loss(pos_scores, neg_scores)
            
        return batch_loss
    
    def compute_loss(self, pos_scores, neg_scores, embeddings=None): ##BPR loss
        # print(pos_scores, neg_scores)
        score_diff = pos_scores - neg_scores

        # Compute the BPR loss
        # print(pos_scores, neg_scores)
        # print(score_diff)
        loss = -torch.log(torch.sigmoid(score_diff) + 1e-10).mean()

        if embeddings:
            lambda_reg = 0.01
            l2_loss = lambda_reg * (
                torch.sum(embeddings[0] ** 2) +
                torch.sum(embeddings[1] ** 2) +
                torch.sum(embeddings[2] ** 2) +
                torch.sum(embeddings[3] ** 2)
            )
            loss += l2_loss

        return loss


    def eval(self, eval_dataset, device, explain=False):
        valid_data, test_data = eval_dataset
        test_len = len(test_data.dataset)

        with torch.no_grad():
            if valid_data:
                ## compute validation loss
                eval_loss = 0
                valid_data = tqdm(valid_data, leave=False)
                valid_data.set_description("Evaluating model loss on validation set")
                for _, (pos_data, neg_data) in enumerate(valid_data):

                    #---for pos data
                    pos_text_input, pos_non_text_input, pos_author_input = pos_data

                    #load data to device
                    pos_text_input = pos_text_input.to(self.device)
                    pos_non_text_input = pos_non_text_input.to(self.device)
                    pos_author_input = pos_author_input.to(self.device)

                    pos_scores, _, _, _, _ = self.forward(pos_text_input, pos_non_text_input, pos_author_input)

                    #---for neg data
                    neg_text_input, neg_non_text_input, neg_author_input = neg_data

                    #load data to device
                    neg_text_input = neg_text_input.to(self.device)
                    neg_non_text_input = neg_non_text_input.to(self.device)
                    neg_author_input = neg_author_input.to(self.device)

                    neg_scores, _, _, _, _ = self.forward(neg_text_input, neg_non_text_input, neg_author_input)

                    eval_loss += self.compute_loss(pos_scores, neg_scores)
            

            ## compute test metrics
            metrics_vals = {}
            total_scores, ys = torch.tensor([]), torch.tensor([])
            total_feature_att_scores, total_title_att_scores, total_text_input = torch.tensor([]).to(device), torch.tensor([]).to(device), torch.tensor([]).to(device)

            test_data = tqdm(test_data, leave=False)
            test_data.set_description("Testing model performance on test set")
            for _, data in enumerate(test_data):
                text_input, non_text_input, user_input, y = data
                text_input = text_input.to(self.device)
                non_text_input = non_text_input.to(self.device)
                user_input = user_input.to(self.device)

                ys = torch.cat((ys, y.cpu().detach()))

                scores, feature_att_score, title_att_score, _, _ = self.forward(text_input, non_text_input, user_input)
                # Calculate the number of elements to set to 1
                scores = scores.cpu().detach()

                # record info in each batch
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

            for e in self.evaluators:
                metrics_vals[repr(e)] = e(ys, preds, test_len) #[1, task]
                

        if explain: #record attention scores for analysis

            # recover text
            tokenizer = BertTokenizer.from_pretrained(self.bert)
            total_title= []
            for token in total_text_input[:,0,:]:
                total_title.append(tokenizer.decode(token))

            feature_list = [
                'month', 
                'IndustryCode1',
                'IndustryCode2',
                'sentiment',
                'topic',
                'eastmoney_robo_journalism',
                'media_robo_journalism',
                'SMA_robo_journalism',
                'item_author_index_rank',
                'article_author_index_rank',
                'article_source_index_rank'
            ]

            report = pd.DataFrame({
                'text': total_title,
                'pred': preds,
                'viral': ys,
                'title_attention': list(total_title_att_scores.cpu().detach().numpy()),
                'features': [feature_list]*(len(total_title)),
                'feature_attention': list(total_feature_att_scores.cpu().detach().numpy())
            })
            
        return eval_loss, metrics_vals, report