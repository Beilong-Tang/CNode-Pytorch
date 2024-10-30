import torch


def braycurtis(y_hat, y, mode="mean"):
    """
    Implement the braycurtis loss:

    Args:
        y_hat: [B,E], the output value
        y:     [B,E], the true value
        mode: string value indicating the loss within the batch:
            mean: the mean value of the loss within the batch
            maximum: the maximum value of the loss within the batch
    Returns:
        torch.Tensor(1): The braycurtis loss
    """
    eps = 1e-8
    abs_diff = torch.abs(y_hat - y).sum(-1)  # [B]
    total_sum: torch.Tensor = (y_hat + y).sum(-1)  # [B]
    res = abs_diff / (total_sum + eps)  # [B]
    if mode == "mean":
        return res.mean()
    elif mode == "maximum":
        return torch.max(res)
    else:
        raise Exception(f"mode {mode} is not supported")


if __name__ == "__main__":
    ## Test braycurtis
    y_hat = torch.tensor([[1, 0, 0], [1, 1, 0]])
    y = torch.tensor([[0, 1, 0], [0, 1, 0]])
    print(braycurtis(y_hat, y))
