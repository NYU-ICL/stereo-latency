from pathlib import Path

import numpy as np
from scipy import stats
import torch
import torch.nn as nn

from invoker import Module


class BaseGazeOffsetModel(Module, nn.Module):
    @classmethod
    def args(cls):
        return {
            "model_root": "./io/model/",
            "in_dims": 2,
            "out_dims": 3,
        }

    def initialize(self):
        self.model_root = Path(self.opt.model_root)
        self.model_root.mkdir(exist_ok=True, parents=True)

    def save(self, fn):
        torch.save(self.state_dict(), self.model_root / fn)

    def load(self, fn):
        self.load_state_dict(torch.load(self.model_root / fn))

    @torch.no_grad()
    def _gen_rv(self, input):
        mean, std, decay = self.forward(input).numpy().T
        rv = stats.exponnorm(decay / std, loc=mean, scale=std)
        return rv

    def pdf_func(self, input):
        rv = self._gen_rv(input)
        return rv.pdf

    def cdf_func(self, input):
        rv = self._gen_rv(input)
        return rv.cdf

    def generate_histogram(self, input, trange, nbins):
        rv = self._gen_rv(input)
        t_bin_edges = np.linspace(*trange, nbins+1)
        t_bin_width = t_bin_edges[1] - t_bin_edges[0]
        cdf = rv.cdf(t_bin_edges[:, None]).mean(axis=-1)
        histogram = np.diff(cdf)
        histogram += 1e-12  # Avoid infinity division for KLDiv
        return histogram

    def generate_quantiles(self, input, label, qrange, nbins):
        rv = self._gen_rv(input)
        cdf = rv.cdf(label.numpy()[:, None]).mean(axis=-1)
        sample_t = np.quantile(label, cdf)

        qrange = np.linspace(*qrange, nbins)
        quantiles = np.quantile(sample_t, qrange)
        return quantiles
