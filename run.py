import argparse
from torchvision import transforms
from datasets.dataset import *
from models.model import *
import torch
import torch.optim as optim 
import numpy as np
import wandb


def main(args): 
    print("learning rate: ", args.lr, "\tnum epochs: ", args.num_epochs, "\n")

    if args.gpu == -1:
        print('disable cuda. ')
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'{device} is available. \n')

    wandb_table = wandb.Table(columns=["repeat", "loss", "acc"])
    
    torchvision_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.Grayscale(num_output_channels=1)
    ])

    test_loss, test_acc = [], [] # list of the test results for 5 repeats
    for i in range(5):
        print("=> {}-th iteration".format(i))
        seed = random.randint(0, 100)
        train_set = KTHDataset(args.dataset_dir, add_reg=args.add_reg, type="train", transform = torchvision_transform, frames = args.frame, seed=seed, device=device)
        test_set = KTHDataset(args.dataset_dir, add_reg=args.add_reg, type="test", transform = torchvision_transform, frames = args.frame, seed=seed, device=device)

        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=True) 
        test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)

        model = Original_Model(mode='KTH', add_reg=args.add_reg).to(device)
        if args.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        best_acc = 0
        for epoch in range(args.num_epochs):
            train_loss, train_acc = train(train_loader, model, optimizer, criterion, device)
            best_acc = max(train_acc, best_acc)
            
            print("Epoch {:3d} : Loss = {:7f}, Acc = {:4f}".format(epoch, train_loss, train_acc))
            # log metrics to wandb
            wandb.log({"train loss": train_loss, "train acc.": train_acc, "best acc.": best_acc})
            
        model.eval()  # test case 학습 방지를 위해.

        with torch.no_grad(): 
            loss, acc = test(test_loader, model, criterion, device)     
            test_loss.append(loss)
            test_acc.append(acc)
            
            print('{}-th Test set: Loss = {:.7f}, Acc = {:.4f}\n'.format(i, loss, acc)) 
            # log metrics to wandb
            wandb.log({"test loss": loss, "test acc.": acc}, step=i)#correct / len(test_loader.dataset)})
            wandb_table.add_data(str(i), loss, acc)
    
    # average the test results
    avg_loss, avg_acc = np.mean(np.array(test_loss)), np.mean(np.array(test_acc))
    wandb.log({"avg test loss": avg_loss, "avg test acc.": avg_acc})
    wandb.log({"Test Acc": wandb.plot.bar( wandb_table, 
                                           label="repeat", value="acc", title="Test Acc Bar Chart")})
    
    print('\nFinal Test Result: \nLoss = {:.7f}, Acc = {:.4f} ({:2.1f}%)\n'.format(avg_loss, avg_acc, avg_acc*100.))

    
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
        output = model(data, aux_data) # forward        # probability = softmax(logit)
        loss = criterion(output, target) 
        loss.backward()  # 역전파
        optimizer.step()
        
        # for monitoring
        train_loss += loss.item() 
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        train_acc += pred.eq(target.view_as(pred)).sum().item()
    
    train_acc /= len(train_loader.dataset) # average
    return train_loss, train_acc


def test(test_loader, model, criterion, device):
    """
    test (go through the model on test_dataset) and return loss and accruacy value.
    """
    test_loss = 0
    test_acc = 0 # best accuracy
    for (data, aux_data), target in test_loader:
        data = data.to(device)
        if args.add_reg :
            aux_data = aux_data.to(device)
        target = target.to(device)
        target = target.to(device)
        output = model(data, aux_data)
        test_loss += criterion(output, target).item() # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        test_acc += pred.eq(target.view_as(pred)).sum().item()
    
    test_acc /= len(test_loader.dataset) # average
    return test_loss, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single Frame ConvNet")
    parser.add_argument("--add_reg", action="store_true")
    parser.add_argument("--dataset_dir", type=str, default="default",
                        help="directory to dataset under 'datasets' folder. default: 'kth-data-aux' if --add_reg else 'kth-data'.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size for training (default: 16)")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="number of epochs to train (default: 30)")
    # parser.add_argument("--start_epoch", type=int, default=1,
    #                     help="start index of epoch (default: 1)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate for training (default: 0.0005)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay for training with SGD optimizer (default: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="weight decay for training with SGD optimizer (default: 0.9)")
    parser.add_argument("--optim", type=str, default="adam",
                        help="optimizer for training (choose one of 'adam' or 'sgd')")
    # parser.add_argument("--log", type=int, default=10,
    #                     help="log frequency (default: 10 iterations)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="set gpu rank to run cuda (set -1 to use cpu only)")
    parser.add_argument("--width", type=int, default = 60)
    parser.add_argument("--height", type=int, default = 80)
    parser.add_argument("--frame", type=int, default = 9,
                        help= "number of consecutive frames as input. choose one of 7 or 9(default).")

    args = parser.parse_args()
    if args.dataset_dir == 'default':
        args.dataset_dir = 'kth-data-aux' if args.add_reg else 'kth-data'
        
    wandb.init(  # TODO
        # set the wandb project where this run will be logged
        project="CS570",
        # track hyperparameters and run metadata
        config={
        "architecture": "3D CNN",
        "dataset": "KTH",
        "optimizer": args.optim,
        "epochs": args.num_epochs,
        "learning_rate": args.lr,
        }
    )
    wandb.run.name = f'{args.optim}-lr-{args.lr}-NE-{args.num_epochs}'
    if args.add_reg:
        wandb.run.name += '-reg'
    main(args)
    wandb.finish()