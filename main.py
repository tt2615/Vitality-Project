

from dataset import PostData, ToTensor#, Log, random_split
from model_temps.lr import LR
from model_temps.llr import LLR
from model_temps.bert import Bert
from model_temps.bertatt import BertAtt

import torch
import atexit
from torch.utils.data import DataLoader, random_split
torch.manual_seed(666)
import argparse
import logging
from tqdm import tqdm, trange
import time
import numpy as np
import re

import warnings
warnings.filterwarnings("ignore")


#Parsing the arguments that are passed in the command line.
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['LR', 'LLR', 'Bert', 'BertAtt'], help="MTL model", required=True)
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
args = parser.parse_args()

#Configure logging
LOG_PATH = (f"./logs/{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.log")
logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.DEBUG, format='%(levelname)s - %(message)s')

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
x_trans_list = [ToTensor()]
y_trasn_list = [ToTensor()] #, Log()
if args.model=='Bert' or args.model=='BertAtt':
    data = PostData(cat_cols = ['stock_code', 'item_author', 'article_author', 'article_source', 'month', 'year', 'eastmoney_robo_journalism', 'media_robo_journalism', 'SMA_robo_journalism'],\
                    num_cols=[],\
                    tar_cols=['viral'],\
                    max_padding_len=args.pad_len,
                    x_transforms=x_trans_list,\
                    y_transforms=y_trasn_list,
                    bert = args.bert)
else:
    data = None #LR to be replaced

gen = torch.Generator()
gen.manual_seed(666)
train_data, valid_data, test_data = random_split(data, [0.8,0.1,0.1], generator=gen) #train:valid:test = 8:1:1
# print(train_data[10])
# exit()

train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=args.batch, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=args.batch, shuffle=True)
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
    model = BertAtt(dim=args.dim, 
                    cat_unique_count=cat_unique_count, 
                    embed_cols_count=embed_feature_count, 
                    num_cols_count=num_feature_count,
                    device=device,
                    bert=args.bert).to(device)
else:
    print('None existing model!')
    exit()
print(f"Model loaded: {args.model}")

# save model before exit
def exit_handler():
    MODEL_PATH = (f"./models/{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.pt")
    torch.save(model.state_dict(), MODEL_PATH)
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
    print("-"*10 + "Start testing" + "-"*10)
    time_s = time.time()
    
    test_loss, metrics, report = model.eval(test_dataloader, device, explain=True)

    if report is not None:
            report.to_csv(f"./analysis/test_{args.model}_{args.batch}_{args.lr}_{args.dim}_{args.optim}_{args.comment}.csv")

    # print result
    print(f"AVG TEST LOSS: {test_loss}")
    for e, val in metrics.items():
        print(f"AVG SCORE for {e}: {val}")
    
    print(f"evalution time {time.time()-time_s}s")
    print("="*10 + "END PROGRAM" + "="*10)

### Train Mode
else: 
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
        
        t_batch = tqdm(train_dataloader, leave=False)
        batch_loss = 0
        for batch, (text_input, non_text_input, y) in enumerate(t_batch):
            # print(x[10])
            if batch%10 == 0:
                t_batch.set_description(f"Batch {batch} - avg loss {batch_loss}")
                t_batch.refresh()
                # logging.info(f"Batch {batch} - avg loss {batch_loss}\n")

            #load data to device
            text_input = text_input.to(device)
            non_text_input = non_text_input.to(device)
            y = y.squeeze().to(torch.long).to(device)

            # print('-----')
            # print(text_input.get_device())
            # print(non_text_input.get_device())
            # print(y.get_device())
            # print(next(model.parameters()).device)

            output = model(text_input, non_text_input)
            pred = output[0]

            batch_loss = model.compute_loss(pred, y)
            # if batch_loss > 0.1: #check if prediction on gt is bad
            #     print(batch_loss, pred.max(1).indices-y)
            epoch_loss += batch_loss
            

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # torch.nn.utils.clip_grad_norm(parameters=model.parameters(), max_norm=10, norm_type=2.0)

            time.sleep(0.01)

        # eavluate on test data
        train_loss = epoch_loss/len(train_dataloader)
        valid_loss, metrics, report = model.eval(valid_dataloader, device, explain=True)
        logging.debug(f"train loss: {train_loss}\n")
        logging.debug(f"valid loss: {valid_loss}\n")
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
    
    
    









