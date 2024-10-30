import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
from sklearn.utils import shuffle
from sklearn.model_selection import KFold

from loss import braycurtis
from data import import_data
import argparse

# Dataset names, learning rates, epochs, and minibatch sizes
DATASETS = ["Drosophila_Gut", "Soil_Vitro", "Human_Gut", "Soil_Vivo"]
LEARNING_RATES = [[0.001, 0.01], [0.01, 0.05], [0.01, 0.05], [0.05, 0.1]]
EPOCHS_LIST = [200, 100, 150, 50]
MINIBATCHES = [5, 23, 5, 25]


# FitnessLayer Class
class FitnessLayer(nn.Module):
    def __init__(self, E):
        super(FitnessLayer, self).__init__()
        self.W = nn.Parameter(torch.randn(E, E))  # Initialize the weight matrix

    def forward(self, p):
        ## p stands for the
        f = self.W @ p
        p_dot = p * (f - torch.ones(p.size(0), 1) @ (p.T @ f))
        return p_dot


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


def train(z, p, model, device): 

    pass


def main(args):
    data = args.data
    assert data in DATASETS
    idx = DATASETS.index(data)
    print(f"Training data {data}")
    z, p = import_data(data, root = args.root) # numpy [N,E], [N,E]
    N, E = z.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FitnessLayer(E).to(device)
    epoch = EPOCHS_LIST[idx]
    
    print(f"training on device: {device}")
    train(z,p,model,device)


    

    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="Human_Gut")
    parser.add_argument("--root", type=str, default=".")
    args = parser.parse_args()
    main(args)
    pass

# Training Reptile loop
def train_reptile(node, epochs, mb, LR, Z, P, report):
    optimizer = optim.Adam(node.parameters(), lr=LR[0])
    for e in range(epochs):
        Z_shuffled, P_shuffled = shuffle(Z, P)
        V = [param.clone() for param in node.parameters()]

        for i in range(0, len(Z_shuffled), mb):
            z_batch = Z_shuffled[i : i + mb]
            p_batch = P_shuffled[i : i + mb]

            optimizer.zero_grad()
            loss = loss(node, z_batch, p_batch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for w, v in zip(node.parameters(), V):
                dv = (w.data - v).clone()
                w.data.copy_(v + dv * LR[1])

        if e % report == 0:
            print(f"Epoch {e}:\tLoss: {loss(node, Z, P)}")

    return node.state_dict()


# Early stopping and training
def train_reptile_early_stop(
    node,
    epochs,
    mb,
    LR,
    Ztrn,
    Ptrn,
    Zval,
    Pval,
    Ztst,
    Ptst,
    report,
    early_stopping_threshold,
):
    optimizer = optim.Adam(node.parameters(), lr=LR[0])
    es_counter = 0

    loss_train = []
    loss_val = []
    loss_test = []

    for e in range(epochs):
        Z_shuffled, P_shuffled = shuffle(Ztrn, Ptrn)
        V = [param.clone() for param in node.parameters()]

        for i in range(0, len(Z_shuffled), mb):
            z_batch = Z_shuffled[i : i + mb]
            p_batch = P_shuffled[i : i + mb]

            optimizer.zero_grad()
            loss = loss(node, z_batch, p_batch)  # [B,N], #[B,N]
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            for w, v in zip(node.parameters(), V):
                dv = (w.data - v).clone()
                w.data.copy_(v + dv * LR[1])

        # Logging losses
        loss_train.append(loss(node, Ztrn, Ptrn).item())
        loss_val.append(loss(node, Zval, Pval).item())
        loss_test.append(loss(node, Ztst, Ptst).item())

        # Early stopping logic
        if e > early_stopping_threshold:
            if es_counter > 10:
                print(f"Early stopping at epoch {e}")
                break

            if loss_val[-1] > loss_val[-2]:
                es_counter += 1
            else:
                es_counter = 0

        if e % report == 0:
            print(
                f"Epoch {e}: Train Loss: {loss_train[-1]}, Val Loss: {loss_val[-1]}, Test Loss: {loss_test[-1]}"
            )

    return node.state_dict(), loss_train, loss_val, loss_test


# Example Usage
N = 10  # Example size
epochs = 100
mb = 32
LR = [1e-3, 1e-3]
Ztrn = torch.rand(100, N)  # Example data
Ptrn = torch.rand(100, N)  # Example data
Zval = torch.rand(20, N)  # Example validation data
Pval = torch.rand(20, N)  # Example validation data
Ztst = torch.rand(20, N)  # Example test data
Ptst = torch.rand(20, N)  # Example test data

node = FitnessLayer(N)

# Train with early stopping
train_reptile_early_stop(
    node,
    epochs,
    mb,
    LR,
    Ztrn,
    Ptrn,
    Zval,
    Pval,
    Ztst,
    Ptst,
    report=10,
    early_stopping_threshold=5,
)
