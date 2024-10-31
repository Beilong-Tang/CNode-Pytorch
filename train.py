import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.model_selection import KFold
import numpy as np
import argparse
import time
import os

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
    name: str,
    device: str,
    epoch: int,
    batch_size: int,
    lr: float,
    reptile_lr: float,
    log,
    log_interval: int,
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

    test_loss_list = []
    train_loss_list = []

    N, E = z.shape

    kf = KFold(n_splits=N)

    for idx, (train_idx, test_idx) in enumerate(kf.split(z)):
        ##
        z_train, p_train = z[train_idx, :], p[train_idx, :]
        z_test, p_test = z[test_idx, :], p[test_idx, :]
        z_train, z_test, p_train, p_test = (
            torch.from_numpy(z_train).float().to(device),
            torch.from_numpy(z_test).float().to(device),
            torch.from_numpy(p_train).float().to(device),
            torch.from_numpy(p_test).float().to(device),
        )
        model = FitnessLayer(E).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        ##
        for e in range(0, epoch):
            start_time = time.time()
            model.train()
            for b_start in range(0, z_train.shape[0], batch_size):
                b_end = min(b_start + batch_size, z_train.shape[0])
                z_train_batch = z_train[b_start:b_end, :]  # [B,E]
                p_train_batch = p_train[b_start:b_end, :]  # [B,E]
                loss_ = loss(model, z_train_batch, p_train_batch)
                loss_.backward()
                optimizer.step()
                optimizer.zero_grad()
            # Reptile update
            for param in model.parameters():
                param.data += reptile_lr * (param.data - param.detach())
            # log
            if e % log_interval == 0:
                log.info(
                    f"Epoch: {e:05} | Tr Loss: {loss_.item():.3f} | Time/Epoch: {int(time.time()-start_time):04}s"
                )
        model.eval()
        with torch.no_grad():
            test_loss = loss(model, z_test, p_test).item()
        log.info(f"Fold {idx}, Test Loss: {test_loss}")
        train_loss_list.append(loss_.item())
        test_loss_list.append(test_loss)


    os.makedirs(f"results/{name}", exist_ok= True)
    np.savetxt(
        f"results/{name}/train_loss.csv", np.array(train_loss_list), delimiter=","
    )
    np.savetxt(f"results/{name}/test_loss.csv", np.array(test_loss_list), delimiter=",")


def main(args):
    data = args.data
    assert data in DATASETS
    idx = DATASETS.index(data)
    log = setup_logger("log", name=data)
    log.info(args)
    log.info(f"Running Experiments {data}")
    z, p = import_data(data, root=args.root)  # numpy [N,E], [N,E]
    log.info(f"total data shape: {z.shape}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epoch = EPOCHS_LIST[idx]
    batch_size = MINIBATCHES[idx]
    reptile_lr = LEARNING_RATES[idx][1]
    lr = LEARNING_RATES[idx][0]
    log.info(f"training on device: {device}")
    os.makedirs("results", exist_ok=True)
    train(z, p, data, device, epoch, batch_size, lr, reptile_lr, log, args.log_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Human_Gut")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument(
        "--log_interval", type=int, default=5, help="the logging interval"
    )
    args = parser.parse_args()
    main(args)
    pass
