#!/usr/bin/env python
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from invoker import Script


class JointModelBase(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {
        }

    @classmethod
    def modules(cls):
        return {
            # Add module dependencies
            "gaze_offset_model": "base",
            "gaze_offset_data_loader": "base",
        }

    @classmethod
    def build_config(cls, args):
        # Args post-processing prior to script main exec
        args.update({
        })
        return args



        #@torch.no_grad()
        #def plot_vergence(c_id):
        #    cmask = (df["c_id"] == c_id).to_numpy()
        #    out = model_vergence(all_input[cmask])
        #    out = out.T[:, None, :].numpy()
        #    t = np.linspace(0, 0.7, 100)[:, None]
        #    pdf = getExpGaussPDF(t, out).mean(axis=-1)
        #    plt.figure(f"Vergence {c_id}")
        #    plt.plot(t, pdf)
        #    plt.hist(df["offset_time"][cmask], density=True, bins=40)
        #    plt.xlim(0, 0.7)

        #def exp_pdf(t, rate):
        #    t[t < 0] = 0
        #    return rate * np.exp(-rate * t)

        #@torch.no_grad()
        #def plot_saccade(c_id):
        #    cmask = (df["c_id"] == c_id).to_numpy()
        #    out = model_saccade(all_input[cmask])
        #    t = np.linspace(0, 0.7, 100)[:, None]
        #    pdf = exp_pdf(t, out.rate.T).mean(axis=-1)
        #    plt.figure(f"Saccade {c_id}")
        #    plt.plot(t, pdf)
        #    plt.hist(df["offset_time"][cmask], density=True, bins=40)
        #    plt.xlim(0, 0.7)

        #@torch.no_grad()
        #def plot_combined(c_id):
        #    cmask = (df["c_id"] == c_id).to_numpy()
        #    out = model_combined(all_input[cmask], model_vergence)
        #    out = out.T[:, None, :].numpy()
        #    t = np.linspace(0, 0.7, 100)[:, None]
        #    pdf = getCombinedPDF(t, out).mean(axis=-1)
        #    plt.figure(f"Combined {c_id}")
        #    plt.plot(t, pdf)
        #    plt.hist(df["offset_time"][cmask], density=True, bins=40)
        #    plt.xlim(0, 0.7)

    def run(self):
        raise RuntimeError("This is an interface for scripts loading the Joint Model!")


if __name__ == "__main__":
    JointModelBase().initialize().run()
