import warnings
warnings.filterwarnings('ignore')

from dataset import PostData, ToTensor
from models import LR

import torch
torch.manual_seed(666)
from torch.utils.data import DataLoader, random_split
import argparse
import logging
from tqdm import tqdm, trange
import time
import atexit

#Parsing the arguments that are passed in the command line.
parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['LR', 'LR-Lasso'], help="MTL model", required=True)
# parser.add_argument('--onehot', action='store_true', help="if data use onehot encoding", required=False)
parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'], default='cpu', help="hardware to perform training", required=False)
parser.add_argument('--model_path', type=str, default=None, help="trained model path", required=False)
parser.add_argument('--batch', type=int, default=64, help="batch size for feeding data", required=False)
parser.add_argument('--lr', type=float, default=1e-3, help="learning rate for training model", required=False)
parser.add_argument('--optim', choices=['SGD', 'ADA'], default="SGD", help="optimizer for training model", required=False)
parser.add_argument('--epoch', type=int, default=100, help="epoch number for training model", required=False)
args = parser.parse_args()

#Configure logging
LOG_PATH = (f"./logs/{args.model}_{args.batch}_{args.lr}_{args.optim}.log")
logging.basicConfig(filename=LOG_PATH, filemode='w', level=logging.DEBUG, format='%(levelname)s - %(message)s')

print("="*20 + "START PROGRAM" + "="*20)

#1. Check device
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
elif args.device == 'mps' and torch.backends.mps.is_available(): # type: ignore
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Computing device: {device}")

#2. Load data
x_trans_list = [ToTensor()]
y_trasn_list = [ToTensor()]
data = PostData(onehot_cols=['sentiment'], tar_cols=['Item_Views', 'Item_Likes', 'Item_Comments'], \
                x_transforms=x_trans_list, y_transforms=y_trasn_list)
train_data, valid_data, test_data = random_split(data, [0.8,0.0,0.2]) #train:test = 8:2
train_dataloader = DataLoader(train_data, batch_size=args.batch, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=len(test_data), shuffle=True)
print(f"Data loaded. Training data: {len(train_data)}; Testing data: {len(test_data)}")


#3. Select model
if args.model == 'LR':
    model = LR(data.get_feature_num(), 3).to(device)
    # print(next(model.parameters()).device)
    
else: # default is LR
    model = LR(data.get_feature_num(), 3).to(device)
print(f"Model loaded: {args.model}")

# save model before exit
def exit_handler():
    MODEL_PATH = (f"./models/{args.model}_{args.batch}_{args.lr}_{args.optim}_{args.epoch}.pt")
    torch.save(model.state_dict(), MODEL_PATH)
atexit.register(exit_handler)

#4. Select optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

### Test Mode
if args.model_path:
    print("-"*10 + "Start evaluating " + "-"*10)

    model = LR(data.get_feature_num(), 3)
    model.load_state_dict(torch.load(f"./models/{args.model_path}"))
    model.to(device)

    test_loss, metrics = model.eval(test_dataloader, device)

    # print result
    print(f"AVG TEST LOSS: {test_loss}")
    for e, val in metrics.items():
        print(f"AVG SCORE for {e}: {val}")
    
    print("="*10 + "END PROGRAM" + "="*10)

### Train Mode
else: 
    print("-"*10 + "Start training " + "-"*10)

    t_epoch = trange(args.epoch, leave=False)
    epoch_loss = 0
    for epoch in t_epoch:
        logging.debug(f"EPOCH {epoch}\n")
        t_epoch.set_description(f"Epoch {epoch} - avg loss: {epoch_loss/len(train_dataloader)}")
        t_epoch.refresh()
        
        t_batch = tqdm(train_dataloader, leave=False)
        batch_loss = 0
        for batch, (x, y) in enumerate(t_batch):
            # print(x[10])
            if batch%10 == 0:
                t_batch.set_description(f"Batch {batch} - avg loss {batch_loss}")
                t_batch.refresh()
                logging.debug(f"Batch {batch} - avg loss {batch_loss}\n")

            #load data to device
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            # print(x.shape,y.shape, pred.shape) [256, 9] [256, 3] [256, 3] 
            
            batch_loss = model.compute_loss(pred, y)
            epoch_loss += batch_loss

            # backpropagation
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # early stop

            time.sleep(0.01)

        # eavluate on test data
        test_loss, metrics = model.eval(test_dataloader, device)
        logging.debug(f"train loss: {epoch_loss/len(train_dataloader)}\n")
        logging.debug(f"test loss: {test_loss}\n")
        logging.debug(f"metrics performance: {metrics}\n")
        logging.debug('-'*10+'\n')
    
    print("="*10 + "END PROGRAM" + "=*10")
    
    
    









