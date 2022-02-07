import pandas as pd
import numpy as np
import torch


pf = pd.read_csv('./offline_592_1000.csv')


def get_sample(initial_data, column, alpha) -> torch.Tensor:
    frequency = pf[column].value_counts(sort=False, normalize=True).tolist()
    try:
        shape = list(torch.squeeze(initial_data, 1).shape) if len(list(initial_data.shape)) > 1 else list(
            initial_data.shape)
        data = list(np.random.choice(range(len(frequency)), p=frequency, size=shape))
        return alpha * torch.Tensor(data).expand(tuple(initial_data.shape)) + (1 - alpha) * initial_data
    except:
        return initial_data

