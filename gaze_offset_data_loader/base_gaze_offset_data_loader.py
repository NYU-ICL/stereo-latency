import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from invoker import Module
from util.pytorch import series2tensor


class BaseGazeOffsetDataLoader(Module):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {
            "data_path": "./io/data/pilot_data_final.csv",
        }

    def initialize(self):
        self.load()

    def load(self):
        self.data_path = Path(self.opt.data_path)
        self.df = pd.read_csv(self.data_path, index_col=0)

        VERGENCE_ANGLE_FIX_FACTOR = 2 # Using angle between both eyes
        self.df["vergence_ang"] *= VERGENCE_ANGLE_FIX_FACTOR

        SECONDS_TO_MILLISECONDS = 1000
        self.df["offset_time"] *= SECONDS_TO_MILLISECONDS

    def estimate_exp_gauss_params(self, aggregation_keys):
        def exponnorm_fit(arr):
            K, loc, scale = stats.exponnorm.fit(offset_times)
            mean = loc
            std = scale
            decay = K * std
            param = np.array([mean, std, decay])
            return param

        input_arr = []
        param_arr = []
        agg_df = self.df.groupby(aggregation_keys).size().reset_index()
        for _, row in agg_df.iterrows():
            condition = row[aggregation_keys].to_numpy()
            masks = [self.df[key] == condition[idx] for idx, key in enumerate(aggregation_keys)]
            offset_times = self.df[np.all(masks, axis=0)]["offset_time"].to_numpy()
            estimation = exponnorm_fit(offset_times)
            input_arr.append(condition)
            param_arr.append(estimation)
        input = torch.tensor(np.stack(input_arr, axis=0, dtype=np.float32))
        param = torch.tensor(np.stack(param_arr, axis=0, dtype=np.float32))
        return input, param

    def generate_df(self):
        return self.df.copy()

    @classmethod
    def generate_tensors(self, df, feature_columns):
        input = series2tensor(df[feature_columns])
        label = series2tensor(df["offset_time"])
        return input, label

    @classmethod
    def generate_histogram(cls, df, trange, nbins):
        t_bin_edges = np.linspace(*trange, nbins+1)
        t_bin_width = t_bin_edges[1] - t_bin_edges[0]
        pdf, _ = np.histogram(df["offset_time"], bins=t_bin_edges, density=True)
        histogram = pdf * t_bin_width
        return histogram

    @classmethod
    def generate_quantiles(cls, df, qrange, nbins):
        qrange = np.linspace(*qrange, nbins)
        quantiles = df["offset_time"].quantile(qrange).to_numpy()
        return quantiles

    @classmethod
    def get_color(cls, cid):
        vrg_color = np.array([251/255,  86/255,   9/255])
        sac_color = np.array([ 56/255, 132/255, 255/255])
        CID_TO_WEIGHTS = [
            (0/2, 1/3), (0/2, 2/3), (0/2, 3/3), (0/2, 1/3), (0/2, 2/3), (0/2, 3/3), (1/2, 1/3),
            (1/2, 2/3), (1/2, 3/3), (2/2, 1/3), (2/2, 2/3), (2/2, 3/3), (1/2, 1/3), (1/2, 2/3),
            (1/2, 3/3), (2/2, 1/3), (2/2, 2/3), (2/2, 3/3), (1/2, 0/3), (2/2, 0/3), (1/2, 0/3),
            (2/2, 0/3),
        ]
        def color_between(vrg_w, sac_w):
            weight_vec = np.array([vrg_w, sac_w])
            weight_vec /= np.linalg.norm(weight_vec)
            return (np.stack([vrg_color, sac_color], axis=-1) ** 2 @ weight_vec) ** 0.5
        color = color_between(*CID_TO_WEIGHTS[cid])
        return color

