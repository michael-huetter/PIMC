import numpy as np
import torch
from dataset import scale, inverse_scale

from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from matplotlib import pyplot as plt

model = torch.load("pip_nn.pth")
device = torch.device("cpu")
model.to(device)
print("Used device:", device)
print("Used model:", model)

# -----------------------------------------------------

X_train_tensor = torch.from_numpy(np.loadtxt("X_train.txt")).reshape(-1, 1)
X_valid_tensor = torch.from_numpy(np.loadtxt("X_valid.txt")).reshape(-1, 1)
Y_train_tensor = torch.from_numpy(np.loadtxt("Y_train.txt"))
Y_valid_tensor = torch.from_numpy(np.loadtxt("Y_valid.txt"))

X_min_tensor = torch.from_numpy(np.loadtxt("X_min.txt"))
X_max_tensor = torch.from_numpy(np.loadtxt("X_max.txt"))

X_scaled_train_tensor = scale(X_train_tensor, X_min_tensor, X_max_tensor)
X_scaled_valid_tensor = scale(X_valid_tensor, X_min_tensor, X_max_tensor)

Y_min_tensor = torch.from_numpy(np.loadtxt("Y_min.txt"))
Y_max_tensor = torch.from_numpy(np.loadtxt("Y_max.txt"))

Y_scaled_train_tensor = scale(Y_train_tensor, Y_min_tensor, Y_max_tensor)
Y_scaled_valid_tensor = scale(Y_valid_tensor, Y_min_tensor, Y_max_tensor)

Y_scaled_train_pred_tensor = model(X_scaled_train_tensor.float())
Y_scaled_valid_pred_tensor = model(X_scaled_valid_tensor.float())

Y_pred_train_tensor = inverse_scale(Y_scaled_train_pred_tensor, Y_min_tensor, Y_max_tensor)
Y_pred_valid_tensor = inverse_scale(Y_scaled_valid_pred_tensor, Y_min_tensor, Y_max_tensor)


# concatenate E_train and E_valid as E_total
E_train = Y_train_tensor.detach().numpy()
E_valid = Y_valid_tensor.detach().numpy()

E_pred_train = Y_pred_train_tensor.detach().numpy()
E_pred_valid = Y_pred_valid_tensor.detach().numpy()

E_total = np.concatenate((E_train, E_valid))
print(E_total)

E_pred_total = np.concatenate((E_pred_train, E_pred_valid))

MAE_train = mean_absolute_error(E_train, E_pred_train)
MAE_valid = mean_absolute_error(E_valid, E_pred_valid)
MAE_total = mean_absolute_error(E_total, E_pred_total)

MSE_train = mean_squared_error(E_train, E_pred_train)
MSE_valid = mean_squared_error(E_valid, E_pred_valid)
MSE_total = mean_squared_error(E_total, E_pred_total)

RMSE_train = np.sqrt(MSE_train)
RMSE_valid = np.sqrt(MSE_valid)
RMSE_total = np.sqrt(MSE_total)

df_err = pd.DataFrame({
    "Dataset": [
        "train",
        "valid",
        "total"
    ],
    "MAE": [
        MAE_train,
        MAE_valid,
        MAE_total
    ],
    "MSE": [
        MSE_train,
        MSE_valid,
        MSE_total
    ],
    "RMSE": [
        RMSE_train,
        RMSE_valid,
        RMSE_total
    ]
})

df_err.to_csv("PIP-NN-Error.csv")

E_err_total = E_pred_total - E_total
print(E_total)

plt.scatter(E_total[:,0], E_err_total[:,0], label="E0")
plt.scatter(E_total[:,1], E_err_total[:,1], label="E1")
plt.xlabel("E")
plt.ylabel("E_err")
plt.legend()
plt.show()