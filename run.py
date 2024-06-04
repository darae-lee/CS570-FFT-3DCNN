import argparse
from torchvision import transforms
from datasets.dataset import *
from models.model import *
import torch
import torch.optim as optim 
import numpy as np
import wandb
from torchsummary import summary
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random

def main(args): 
    print("learning rate: ", args.lr, "\tnum epochs: ", args.num_epochs, 
          "\tregularization: ", args.add_reg, "\tlambda: "+ str(args.lbda)+"\n" if args.add_reg else "\n")

    if args.gpu == -1:
        print('disable cuda. ')
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available. \n')

    wandb_table = wandb.Table(columns=["repeat", "time", "loss", "acc", "auc"])
    
    torchvision_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.Grayscale(num_output_channels=1)
    ])

    test_loss, test_acc, test_auc = [], [], [] # list of the test results for 5 repeats
    train_time = []
    total_iteration = 1
    for i in range(total_iteration):
        print("=> {}-th iteration".format(i))
        seed = args.seed + i
        random.seed(seed)
        train_set = KTHDataset(args.dataset_dir, 
                                fft=args.fft,
                                cut_param=args.cut_param,
                                add_reg=args.add_reg, 
                                type="train", 
                                transform = torchvision_transform, 
                                frames = args.frame, 
                                seed=seed, 
                                device=device)
        test_set = KTHDataset(args.dataset_dir, 
                                fft=args.fft,
                                cut_param=args.cut_param,
                                add_reg=args.add_reg, 
                                type="test", 
                                transform = torchvision_transform, 
                                frames = args.frame, 
                                seed=seed, 
                                device=device)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)

        if args.fft == "FFT":
            model = FFT_Model(mode='KTH', cut_param=args.cut_param).to(device)
        elif args.fft == "FFT3":
            model = FFT3_Model(mode='KTH', cut_param=args.cut_param).to(device)
        else:
            model = Original_Model(mode='KTH', add_reg=args.add_reg).to(device)

        if args.fft == "FFT":
            summary(model, input_size = (1, 2*args.frame, args.height, args.width))
        elif args.fft == "FFT3":
            summary(model, input_size = (1, 3*args.frame, args.height, args.width))
        else: 
            summary(model, input_size = (1, 5*args.frame-2, args.height, args.width))

        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        best_acc = 0
        best_epoch = 0
        best_auc = 0
        stop_count = 0
        patience = 50

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for epoch in range(args.num_epochs):
            model.train()
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            
            model.eval()  # Evaluate on the test set
            with torch.no_grad():
                test_epoch_loss, test_epoch_acc, test_epoch_auc = test(test_loader, model, criterion, device)
                
            print("Epoch {:3d} : Train Loss = {:7f}, Train Acc = {:4f}, Test Loss = {:7f}, Test Acc = {:4f}, Test AUC = {:4f}".format(
                epoch, train_loss, train_acc, test_epoch_loss, test_epoch_acc, test_epoch_auc))
            # log metrics to wandb
            wandb.log({"train loss": train_loss, "train acc.": train_acc, 
                       "test loss": test_epoch_loss, "test acc.": test_epoch_acc, "best acc.": best_acc, "test auc": test_epoch_auc, "best auc.": best_auc})
            
            # Early stopping check
            if test_epoch_acc > best_acc:
                best_acc = test_epoch_acc
                best_epoch = epoch
                best_auc = test_epoch_auc
                stop_count = 0
            else:
                stop_count += 1
                
            if stop_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        avg_time = time / epoch
        test_loss.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)
        test_auc.append(test_epoch_auc)
        train_time.append(avg_time)
        print('{}-th Train set: Time = ', avg_time)
        print('{}-th Test set: Loss = {:.7f}, Acc = {:.4f}, AUC = {:.4f}\n'.format(i, test_epoch_loss, test_epoch_acc, test_epoch_auc)) 
        wandb.log({"train time": avg_time, "test loss": test_epoch_loss, "test acc.": test_epoch_acc, "test AUC": test_epoch_auc}, step=i)
        wandb_table.add_data(str(i), avg_time, test_epoch_loss, test_epoch_acc, test_epoch_auc)
    
    # average the test results
    avg_time = np.mean(np.array(train_time))
    avg_loss, avg_acc, avg_auc = np.mean(np.array(test_loss)), np.mean(np.array(test_acc)), np.mean(np.array(test_auc))
    wandb.log({"avg train time": avg_time})
    wandb.log({"avg test loss": avg_loss, "avg test acc": avg_acc, "avg test auc": avg_auc})
    wandb_table.add_data("avg", avg_time, avg_loss, avg_acc, avg_auc)
    wandb.log({"Test Acc": wandb.plot.bar(wandb_table, 
                                           label="repeat", value="acc", title="Test Acc Bar Chart")})
    
    print('\nFinal Test Result: \nLoss = {:.7f}, Acc = {:.4f} ({:2.1f}%), Acc = {:.4f}\n'.format(avg_loss, avg_acc, avg_acc*100., avg_auc))

    
def train(train_loader, model, optimizer, criterion, device):
    """
    train (forward + backward + optimizer update) for 1 epoch 
    and return loss value.
    """
    train_loss = 0
    train_acc = 0 # best acc
    for (data, aux_data), target in train_loader:

        data = data.to(device)
        if args.add_reg :
            aux_data = aux_data.to(device)
        target = target.to(device)
        optimizer.zero_grad()  # 기울기 초기화
        output = model(data) # forward        # probability = softmax(logit)
        if args.add_reg:
            pred_feat = output[:, :aux_data.size(-1)]
            pred_feat = (pred_feat - pred_feat.mean()) / pred_feat.std()
            aux_data = (aux_data - aux_data.mean()) / aux_data.std()
            regularization = torch.nn.functional.mse_loss(pred_feat, aux_data)
        else:
            regularization = 0
        loss = criterion(output[:, -model.classes:], target) + args.lbda * regularization
        loss.backward()  # 역전파
        optimizer.step()
        
        # for monitoring
        train_loss += loss.item() * data.size(0)
        pred = output[:, -model.classes:].argmax(dim=1, keepdim=True) # get the index of the max log-probability
        train_acc += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)  # average
    train_acc /= len(train_loader.dataset) # average
    return train_loss, train_acc


def test(test_loader, model, criterion, device):
    """
    test (go through the model on test_dataset) and return loss and accruacy value.
    """
    test_loss = 0
    test_acc = 0 # best accuracy
    test_auc = 0
    labels = []
    predicts = []

    for (data, aux_data), target in test_loader:
        data = data.to(device)
        if args.add_reg :
            aux_data = aux_data.to(device)
        target = target.to(device)
        output = model(data)
        if args.add_reg:
            pred_feat = output[:, :aux_data.size(-1)]
            pred_feat = (pred_feat - pred_feat.mean()) / pred_feat.std()
            aux_data = (aux_data - aux_data.mean()) / aux_data.std()
            regularization = torch.nn.functional.mse_loss(pred_feat, aux_data)
        else:
            regularization = 0
        loss = criterion(output[:, -model.classes:], target) + args.lbda * regularization
        test_loss += loss.item() * data.size(0) # sum up batch loss
        pred = output[:, -model.classes:].argmax(dim=1, keepdim=True) # get the index of the max log-probability
        test_acc += pred.eq(target.view_as(pred)).sum().item()
        m = nn.Softmax(dim = 1)
        output_softmax = m(output)
        
        predicts = predicts+ output_softmax.detach().cpu().tolist()
        labels = labels+ target.detach().cpu().tolist()
    roc_auc = roc_auc_score(labels, predicts, multi_class='ovr', average="macro", labels=[0,1,2,3,4,5])
    
    test_loss /= len(test_loader.dataset) # average
    test_acc /= len(test_loader.dataset) # average
    test_auc = roc_auc
    return test_loss, test_acc, test_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process dataset and Train model")
    parser.add_argument("--fft", type=str, default=None)
    parser.add_argument("--add_reg", action="store_true")
    parser.add_argument("--cut_param", type=float, default=1.0)
    parser.add_argument("--dataset_dir", type=str, default="default",
                        help="directory to dataset under 'datasets' folder. default: 'kth-data-aux' if --add_reg else 'kth-data'.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for training (default: 32)")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="number of epochs to train (default: 30)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate for training (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay for training with SGD optimizer (default: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="weight decay for training with SGD optimizer (default: 0.9)")
    parser.add_argument("--optim", type=str, default="adam",
                        help="optimizer for training (choose one of 'adam' or 'sgd')")
    parser.add_argument("--gpu", type=int, default=0,
                        help="set gpu rank to run cuda (set -1 to use cpu only)")
    parser.add_argument("--width", type=int, default = 60)
    parser.add_argument("--height", type=int, default = 80)
    parser.add_argument("--frame", type=int, default = 9,
                        help= "number of consecutive frames as input. choose one of 7 or 9(default).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lbda", type=float, default=0.005,
                        help="lambda for regularization (default: 0.005)")

    args = parser.parse_args()
    if args.dataset_dir == 'default':
        args.dataset_dir = 'kth-data-aux' if args.add_reg else 'kth-data'
    arch = "3DCNN"
    if args.fft:
        arch = arch + args.fft
    config = {"architecture": arch, "dataset": "KTH"}
    config_name = {'lr': 'learning_rate', 'optim':'optimizer', 'num_epochs':'epochs'}
    for k in args.__dict__:
        if k in config_name.keys():
            config[config_name[k]] = args.__dict__[k]
        else:
            config[k] = args.__dict__[k]

    wandb.init(  # TODO
        # set the wandb project where this run will be logged
        project="CS570-final", 
        config=config
    )
    wandb.run.name = f'model-{config["architecture"]}-kth-{args.optim}-lr-{args.lr}-NE-{args.num_epochs}-{args.cut_param}' + ('-reg' if args.add_reg else '')
    main(args)
    wandb.finish()