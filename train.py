import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import time

from loss import braycurtis
from data import import_data
from log import setup_logger


# Dataset names, learning rates, epochs, and minibatch sizes
DATASETS = ["Drosophila_Gut", "Soil_Vitro", "Human_Gut", "Soil_Vivo"]
LEARNING_RATES = [[0.001, 0.01], [0.01, 0.05], [0.01, 0.05], [0.05, 0.1]]
EPOCHS_LIST = [200, 100, 150, 50]
MINIBATCHES = [5, 23, 5, 25]


# FitnessLayer Class
class FitnessLayer(nn.Module):
    def __init__(self, E):
        super(FitnessLayer, self).__init__()
        self.W = nn.Parameter(torch.zeros(E, E))  # Initialize the weight matrix

    def forward(self, t, p):
        ## p stands for the
        f = self.W @ p
        f = f.unsqueeze(1)
        p = p.unsqueeze(1)
        p_dot = p * (f - (torch.ones(p.size(0), 1).to(p.device) @ p.T) @ f)
        return p_dot.squeeze(1)


# Neural ODE prediction function
def predict(f, z):
    """
    Args:
        f: the ode function
        z: [B, E], the initial state of the ODE where E stands for the dimension
    Returns:
        y_hat: [B, E], the final output at time step t = 1
    """
    # def node_ode(x):
    #     # Neural ODE using torchdiffeq
    #     return odeint(θ, x, torch.tensor([0.0, 1.0]), method="dopri5")[-1]

    # # Predict for each batch of inputs
    # q = torch.cat([node_ode(z[:, i].unsqueeze(1)) for i in range(z.size(1))], dim=1)
    res = []
    for i in range(z.size(0)):
        res.append(odeint(f, z[i], torch.tensor([0.0, 1.0]), method="dopri5")[-1])
    return torch.stack(res, dim=0)


# Bray-Curtis Loss function
def loss(f, z, y):
    """
    calculate braycurtis loss
    Args:
        f: the ode function
        z: [B, E], the initial state of the ODE where E stands for the dimension
        y: [B, E], the final ground truth at time step t = 1
    Return:
        loss: torch.Tensor(1)
    """
    y_hat = predict(f, z)
    return braycurtis(y_hat, y)


def train(
    z: np.ndarray,
    p: np.ndarray,
    model: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    batch_size: int,
    reptile_lr: float,
    log
):
    """
    z: [N,E]
    p: [N,E]
    model: nn.Module
    optimzer: optim.Optimizer
    device: str
    eopch: int
    batch_size: int
    """
    ## Split z, p into training and testing dataset
    z_train, z_test, y_train, y_test = train_test_split(
        z, p, test_size=0.2, random_state=42
    )
    z_train, z_test, y_train, y_test = (
        torch.from_numpy(z_train).float().to(device),
        torch.from_numpy(z_test).float().to(device),
        torch.from_numpy(y_train).float().to(device),
        torch.from_numpy(y_test).float().to(device),
    )
    for e in range(0, epoch):
        ct = 0
        start_time = time.time()
        model.train()
        for b_start in range(0, z_train.shape[0], batch_size):
            b_end = min(b_start + batch_size, z_train.shape[0])
            z_train_batch = z_train[b_start:b_end, :]  # [B,E]
            y_train_batch = y_train[b_start:b_end, :]  # [B,E]
            loss_ = loss(model, z_train_batch, y_train_batch)
            loss_.backward()
            optimizer.step()
            # Reptile update
            for param in model.parameters():
                param.data += reptile_lr * (param.data - param.detach())
        model.eval()
        with torch.no_grad():
            loss_cv = loss(model, z_test, y_test).item()
        log.info(
            f"Epoch: {e:05} | CV Loss: {loss_cv:.3f} | Time: {int(time.time()-start_time):04}s"
        )


def main(args):
    data = args.data
    assert data in DATASETS
    idx = DATASETS.index(data)
    log = setup_logger("log")
    log.info(f"Training data {data}")
    z, p = import_data(data, root=args.root)  # numpy [N,E], [N,E]
    N, E = z.shape
    log.info(f"total data shape: {z.shape}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FitnessLayer(E).to(device)
    epoch = EPOCHS_LIST[idx]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATES[idx][0])
    batch_size = MINIBATCHES[idx]
    reptile_lr = LEARNING_RATES[idx][1]
    log.info(f"training on device: {device}")
    train(z, p, model, optimizer, device, epoch, batch_size, reptile_lr, log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Human_Gut")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--save_interval", type=int, default=5, help="determines saving interval based on epoch number")
    parser.add_argument("--save_dir", type=str, default="ckpt", help="the folder for saving the ckpt")
    parser.add_argument("--name", type=str, default=None, help="determines the name of the experiment")
    args = parser.parse_args()
    print(args)
    main(args)
    pass