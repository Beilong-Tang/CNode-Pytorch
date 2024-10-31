from typing import Tuple
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
import os


def import_data(data: str, root=".") -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        data: The name of the data
        root: the path root
    Returns:
        Z: numpy [B,E], the initial [1,1,1,0,...] vector after nomalization
        P: numpy [B,E], the final distribution

    """
    P = np.loadtxt(os.path.join(root, f"Data/Experimental/{data}/P.csv" ), delimiter=",")
    P = np.hstack([normalize(P[:, [i]], norm="l1", axis=0) for i in range(P.shape[1])])
    Z = np.where(P > 0, 1, 0)
    Z = np.hstack([normalize(Z[:, [i]], norm="l1", axis=0) for i in range(Z.shape[1])])
    if data == "Soil_Vivo":
        Z, P = shuffle(Z, P)
        Z, P = np.split(Z, [int(0.1 * len(Z))]), np.split(P, [int(0.1 * len(P))])
    Z = np.transpose(Z)
    P = np.transpose(P)
    return Z, P
