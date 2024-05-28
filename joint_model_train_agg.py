#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from invoker import Script


class JointModelTrainAgg(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "uid_removal_mask": "",
            "percent_removal_mask": 0.0,
            "rng_seed": 1,
            "lr": 1e-2,
            "nepochs": 200000,
            "log_freq": 5000,
            "display": False,
            "skip_write": False,
        })
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update({
            # Add module dependencies
            "gaze_offset_criterion": "mse",
            "gaze_offset_data_loader": "base",
            "gaze_offset_model": "rbf",
        })
        return mods

    @classmethod
    def build_config(cls, args):
        # Args post-processing prior to script main exec
        args = super().build_config(args)
        if args["model_label"] == "vergence":
            feature_columns = ["vergence_ang"]
        elif args["model_label"] == "saccade":
            feature_columns = ["saccade_ang"]
        else:
            feature_columns = ["vergence_ang", "saccade_ang"]
        args.update({
            "feature_columns": feature_columns,
            "gaze_offset_model.in_dims": len(feature_columns),
            "gaze_offset_model.rng_seed": args["rng_seed"],
        })
        return args

    def run(self):
        # Init random seed
        torch.manual_seed(self.opt.rng_seed)
        np.random.seed(self.opt.rng_seed)

        # Load Data
        df = self.gaze_offset_data_loader.generate_df()
        if self.opt.uid_removal_mask != "":
            df = df[df.u_id != self.opt.uid_removal_mask]
        if self.opt.percent_removal_mask != 0.0:
            df = df.sample(frac=1 - self.opt.percent_removal_mask)
        self.gaze_offset_data_loader.df = df

        input, label = self.gaze_offset_data_loader.estimate_exp_gauss_params(self.opt.feature_columns)

        # Optimizer
        optimizer = torch.optim.Adam(self.gaze_offset_model.parameters(), lr=self.opt.lr)

        def train_loop(epochi):
            optimizer.zero_grad()
            output = self.gaze_offset_model(input)
            loss = self.gaze_offset_criterion.loss(output, label)
            total_loss = loss.sum()
            total_loss.backward()
            optimizer.step()
            return {
                "mean_loss": loss[0].item(),
                "std_loss": loss[1].item(),
                "decay_loss": loss[2].item(),
            }

        # Training
        loss_arr = []
        for epochi in range(self.opt.nepochs):
            emit_data = train_loop(epochi)
            if (epochi+1) % self.opt.log_freq == 0 or epochi == 0:
                emit_list = [f"{k}: {v:.3f}" for k, v in emit_data.items()]
                logging.info("[%04d] %s", epochi+1, " ".join(emit_list))
                loss_arr.append(emit_data["mean_loss"])

        # Store Model
        fn_suffix = Path(self.gaze_offset_data_loader.opt.data_path).stem
        model_path = f"{self.opt.model_label}_{self.modules()['gaze_offset_model']}_gaze_offset_model_{fn_suffix}.pth"

        if not self.opt.skip_write:
            self.gaze_offset_model.save(model_path)

        # Visualize Training Loss
        if self.opt.display:
            plt.plot(loss_arr)
            plt.show()


if __name__ == "__main__":
    JointModelTrainAgg().initialize().run()
