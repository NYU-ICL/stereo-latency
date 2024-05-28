import torch

def series2tensor(series):
    out = torch.tensor(series.to_numpy(), dtype=torch.float32)
    out = out.reshape(out.shape[0], -1)
    return out

