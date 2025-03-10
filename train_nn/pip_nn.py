import torch
from torch import nn

from torch.utils.tensorboard import SummaryWriter
from time import time
from datetime import timedelta

class PIP_NN(nn.Module):
    def __init__(self, m, n):
        super(PIP_NN, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(m, 10),
            nn.Tanh(),
            nn.Linear(10, 50),
            nn.Tanh(),
            nn.Linear(50, n)
        )

    def forward(self, pip):
        E = self.layer_stack(pip)
        return E
    

def train(train_loader, model, loss_fn, optimizer, device):
        # size of train dataset
        train_size = len(train_loader.dataset)
        # set model to train mode
        model.train()
        # loss
        loss = 0.0

        # start train
        for batch, (X, y) in enumerate(train_loader):
            # move data to target device
            X = X.to(device)
            y = y.to(device)

            # predict and loss
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logs
            if batch % 1000 == 0:
                current = batch * len(X)
                print(f"loss: {loss.item():.7f} [{current:7d} / {train_size:7d}]")

        print(f"loss: {loss.item():.7f} [{train_size:7d} / {train_size:7d}]")

        return loss
    
def valid(valid_loader, model, loss_fn, device):
        # size of valid dataset
        valid_size = len(valid_loader.dataset)
        # number of batches
        n_batches = len(valid_loader)

        # set model to eval mode
        model.eval()

        # total loss
        loss_tot = 0

        with torch.no_grad():
            for (X, y) in valid_loader:
                # to device
                X = X.to(device)
                y = y.to(device)
                
                # loss
                y_pred = model(X)
                loss = loss_fn(y_pred, y)
                loss_tot += loss.item()
        
        loss_tot /= n_batches

        print(f"loss: {loss_tot:.7f} [{valid_size:7d} / {valid_size:7d}]")

        return loss_tot

def train_nn(epoches, optimizer, device, loss_fn, model, train_loader, schedular):
    writer = SummaryWriter("logs/" + str(int(time())))
    start = time()
    for t in range(epoches):
       # start
       print(f"Epoch {t + 1}")
       print("-" * 80)
       epoch_start = time()
       # save learning rate
       current_lr = optimizer.state_dict()["param_groups"][0]["lr"]
       writer.add_scalar("learning_rate", current_lr, t)
       print(f"Learning rate: {current_lr:.7f}")
       # train
       print("Train:")
       train_loss = train(train_loader, model, loss_fn, optimizer, device)
       writer.add_scalar("train_loss", train_loss, t)
       train_end = time()
       print(f"Train time: {timedelta(seconds=(train_end - epoch_start))}")
       # valid
       print("Valid:")
       valid_loss = valid(train_loader, model, loss_fn, device)
       writer.add_scalar("valid_loss", valid_loss, t)
       writer.flush()
       valid_end = time()
       print(f"Valid time: {timedelta(seconds=(valid_end - train_end))}")
       # update learning rate according to loss
       schedular.step(valid_loss)
       
       # total epoch
       epoch_end = time()
       print(f"Epoch time: {timedelta(seconds=(epoch_end - epoch_start))}")
       print()
       end = time()
       print("Done!")
       print(f"Total time: {timedelta(seconds=(end - start))}")
       torch.save(model, f"pip_nn.pth")
       print("Model saved to pip_nn.pth")
