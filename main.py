

from dataset.bertdata import BertData
from dataset.bprdata import BprData
from dataset.transform import ToTensor#, Log, random_split
from model_temps.lr import LR
from model_temps.llr import LLR
from model_temps.bert import Bert
from model_temps.bertatt import BertAtt
from model_temps.bertbpr import BertAttBpr
from evaluator import ACCURACY, CLASSIFICATION

import torch
import atexit
from torch.utils.data import DataLoader, random_split

import argparse
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import re
import os

import warnings
warnings.filterwarnings("ignore")

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.mps.manual_seed(seed)

#Parsing the arguments that are passed in the command line.
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['LR', 'LLR', 'Bert', 'BertAtt', 'BertBpr','BertBpr_v2'], help="MTL model", required=True)
# parser.add_argument('--onehot', action='store_true', help="if data use onehot encoding", required=False)
parser.add_argument('--device', type=str, default='cpu', help="hardware to perform training", required=False)
parser.add_argument('--mode', choices=['train', 'test'], default='train', help="train model or test model", required=False)
parser.add_argument('--model_path', type=str, default=None, help="trained model path", required=False)
parser.add_argument('--batch', type=int, default=64, help="batch size for feeding data", required=False)
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for training model", required=False)
parser.add_argument('--optim', choices=['SGD', 'Adam', 'AdamW'], default="Adam", help="optimizer for training model", required=False)
parser.add_argument('--epoch', type=int, default=100, help="epoch number for training model", required=False)
parser.add_argument('--dim', type=int, default=20, help="dimension for latent factors", required=False)
parser.add_argument('--comment', type=str, help="additional comment for model", required=False)
parser.add_argument('--percent', type=int, default=5, help="mark top percentage as viral", required=False)
parser.add_argument('--pad_len', type=int, default=32, help="maximum padding length for a sentence", required=False)
parser.add_argument('--bert', type=str, default='Langboat/mengzi-bert-base-fin', choices=['bert-base-chinese','Langboat/mengzi-bert-base-fin'], help="version of bert", required=False)
parser.add_argument('--oversample', type=bool, default=False, help="whether oversample viral post", required=False)
parser.add_argument('--report', type=bool, default=True, help="whether generate report", required=False)
args = parser.parse_args()

#Configure logging
LOG_PATH = (f"./logs/{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.log")
logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.DEBUG, format='%(levelname)s - %(message)s')

MODEL_PATH = (f"./models/{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.pt")

print("="*20 + "START PROGRAM" + "="*20)

#1. Configure device
if re.compile('(cuda|cuda:\d+)').match(args.device) and torch.cuda.is_available():
    device = torch.device(args.device)
elif args.device == 'mps' and torch.backends.mps.is_available(): # type: ignore
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Computing device: {device}")

#2. Load data
if args.model=='Bert' or args.model=='BertAtt':
    x_trans_list = [ToTensor()]
    y_trasn_list = [ToTensor()] #, Log()
    data = BertData(cat_cols = ['stock_code', 
                                'item_author_cate', 
                                'article_author', 
                                'article_source_cate', 
                                'month', 
                                'eastmoney_robo_journalism', 
                                'media_robo_journalism', 
                                'SMA_robo_journalism'],\
                    num_cols=['sentiment_score'],\
                    # num_cols=['item_views'],\
                    topic_cols=['topics_val1',
                                'topics_val2',
                                'topics_val3',
                                'topics_val4',
                                'topics_val5'],\
                    tar_cols=['viral'],\
                    max_padding_len=args.pad_len,
                    x_transforms=x_trans_list,\
                    y_transforms=y_trasn_list,
                    bert = args.bert)
    gen = torch.Generator()
    gen.manual_seed(666)
    train_data, valid_data, test_data = random_split(data, [0.8,0.1,0.1], generator=gen) #train:valid:test = 8:1:1
    # print(train_data[0][1])
    # exit()                    

    train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch, shuffle=True)
    valid_dataset = valid_dataloader
    test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=True)    
    test_dataset = test_dataloader         
                                                                                    
elif args.model=='BertBpr':
    x_trans_list = [ToTensor()]
    data = BprData(cat_cols = ['stock_code',
                                'month', 
                                'eastmoney_robo_journalism', 
                                'media_robo_journalism', 
                                'SMA_robo_journalism'],\
                    num_cols=['sentiment_score'],\
                    topic_cols=['topics_val1',
                                'topics_val2',
                                'topics_val3',
                                'topics_val4',
                                'topics_val5',],\
                    user_cols = ['item_author_cate', 
                                'article_author', 
                                'article_source_cate'],\
                    tar_col = 'viral',
                    max_padding_len=args.pad_len,
                    x_transforms=x_trans_list,
                    bert = args.bert)
                                                                   
    train_data = data.train_data
    valid_data = data.valid_data
    test_data = data.test_data

    train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=True)
    valid_dataset = test_dataset = (valid_dataloader, test_dataloader)

elif args.model=='BertBpr_v2':
    x_trans_list = [ToTensor()]
    data = BprData(dir='./data/eastmoney_bert_ind.csv',
                    cat_cols = ['month', 
                               'IndustryCode1',
                               'IndustryCode2'
                                ],\
                    num_cols=['sentiment_score'],\
                    topic_cols=['topics_val1',
                                'topics_val2',
                                'topics_val3',
                                'topics_val4',
                                'topics_val5',],\
                    user_cols = ['eastmoney_robo_journalism', 
                                'media_robo_journalism', 
                                'SMA_robo_journalism'],\
                    tar_col = 'viral',
                    max_padding_len=args.pad_len,
                    x_transforms=x_trans_list,
                    bert = args.bert)
                                                                   
    train_data = data.train_data
    valid_data = data.valid_data
    test_data = data.test_data

    train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=args.batch, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=True)
    valid_dataset = test_dataset = (valid_dataloader, test_dataloader)

print(f"Data loaded. Training data: {len(train_data)}; Valid data: {len(valid_data)}; Testing data: {len(test_data)}")


#3. Select model
if args.model_path:
    model = LR(data.get_feature_num(), data.get_task_num())
    model.load_state_dict(torch.load(f"./models/{args.model_path}.pt"))
    model.to(device)
if args.model == 'LR':
    model = LR(data.get_feature_num(), data.get_task_num()).to(device)
elif args.model == 'LLR':
    model = LLR(data.get_feature_num(), data.get_task_num()).to(device)
elif args.model == 'Bert': # default is Bert
    # count cat unique count for embedding: ['stock_code', 'item_author', 'article_author', 'article_source']
    cat_unique_count = data.get_embed_feature_unique_count()
    embed_feature_count = data.get_embed_feature_count()
    num_feature_count = data.get_num_feature_count()
    model = Bert(args.dim, cat_unique_count, embed_feature_count, num_feature_count,device,args.bert).to(device)
elif args.model == 'BertAtt':
    cat_unique_count = data.get_embed_feature_unique_count()
    embed_feature_count = data.get_embed_feature_count()
    num_feature_count = data.get_num_feature_count()
    topic_num = data.get_topic_num()
    model = BertAtt(dim=args.dim, 
                    cat_unique_count=cat_unique_count, 
                    embed_cols_count=embed_feature_count, 
                    num_cols_count=num_feature_count,
                    topic_num=topic_num,
                    device=device,
                    bert=args.bert).to(device)
elif args.model == 'BertBpr' or 'BertBpr_v2':
    cat_unique_count = data.get_cat_feature_unique_count()
    user_unique_count = data.get_user_feature_unique_count()
    cat_feature_count = data.get_cat_feature_count()
    num_feature_count = data.get_num_feature_count()
    user_feature_count = data.get_user_feature_count()
    topic_num = data.get_topic_num()
    model = BertAttBpr(dim=args.dim, 
                    cat_unique_count=cat_unique_count,
                    user_unique_count=user_unique_count,
                    cat_cols_count=cat_feature_count, 
                    user_cols_count=user_feature_count,
                    num_cols_count=num_feature_count,
                    topic_num=topic_num,
                    device=device,
                    bert=args.bert).to(device)
else:
    print('Invalid model choice!')
    exit()
print(f"Model created: {args.model}")

# save model before exit
def exit_handler():
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"save model to {MODEL_PATH}!")
atexit.register(exit_handler)

#4. Select optimizer
if args.optim=='SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
elif args.optim=='AdamW': # good for transformer based
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
else: # default adam
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

### Test Mode
if args.mode=="test":
    if not os.path.isfile(MODEL_PATH):
        print(f"Exit testing because no model found at {MODEL_PATH}!")
    else:
        # MODEL_PATH = './models/Bert_64_0.001_Adam_None.pt'
        print(f"Model {MODEL_PATH} loaded for testing")
        model.load_state_dict(torch.load(MODEL_PATH))

    print("-"*10 + "Start testing" + "-"*10)
    time_s = time.time()
    
    test_loss, metrics, report = model.eval(test_dataset, device, explain=True)

    # print result
    print(f"AVG TEST LOSS: {test_loss/len(test_dataloader)}")
    for e, val in metrics.items():
        print(f"AVG SCORE for {e}: {val}")

    if report is not None:
        report.to_csv(f"./analysis/test_{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.csv")
    
    print(f"evalution time {time.time()-time_s}s")
    print("="*10 + "END PROGRAM" + "="*10)

### Train Mode
elif args.mode=="train": 
    print("-"*10 + "Start training" + "-"*10)

    # paramters for early stop
    best_loss = np.inf
    min_delta=0.001
    counter = 0
    patience = 10
    stop_training = False

    t_epoch = trange(args.epoch, leave=False)
    epoch_loss = 0
    for epoch in t_epoch:
        logging.debug(f"EPOCH {epoch}\n")
        t_epoch.set_description(f"Epoch {epoch} - avg loss: {epoch_loss/len(train_dataloader)}")
        t_epoch.refresh()
        epoch_loss = 0 #reset epoch loss for current epoch training
        
        batch_loss = 0
        batch_tqdm = tqdm(train_dataloader, leave=False)
        for batch, batch_data in enumerate(batch_tqdm):

            # record batch_loss
            batch_tqdm.set_description(f"Batch {batch} - batch loss {batch_loss}")
            batch_tqdm.refresh()
            # logging.info(f"Batch {batch} - avg loss {batch_loss}\n")
            
            batch_loss = model.train(batch_data)

            epoch_loss += batch_loss

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)

            # time.sleep(0.01)

        # eavluate on test data
        batch_tqdm.set_description(f"Epoch {epoch} evaluation:")
        valid_loss, metrics, report = model.eval(valid_dataset, device, explain=True)
        for e, val in metrics.items():
            print(f"AVG SCORE for {e}: {val}")

        logging.debug(f"train loss: {epoch_loss/len(train_dataloader)}\n")
        logging.debug(f"valid loss: {valid_loss/len(valid_dataloader)}\n")
        logging.debug(f"metrics performance: {metrics}\n")
        logging.debug('-'*10+'\n')

        if report is not None:
            report.to_csv(f"./analysis/valid_{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}_epoch{epoch}.csv")

        # early stop
        # Check for early stopping
        if valid_loss < best_loss - min_delta:
            best_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                logging.debug("Early stopping: validation loss did not improve for {} epochs".format(patience))
                stop_training = True
                break
    
    print("="*10 + "END PROGRAM" + "="*10)
    
    
    









