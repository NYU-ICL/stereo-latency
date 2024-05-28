#!/usr/bin/env python
from itertools import product
import logging

import pandas as pd
import torch
from scipy.special import rel_entr

from joint_model_base import JointModelBase


class JointModelDatasplitAnalysis(JointModelBase):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_fn": "full_rbf_gaze_offset_model_pilot_data_final.pth",
            "output_fn": "io/stats/joint_model_datasplit_kldiv.csv",
            "trange": [0, 1200],
            "nbins": 50,
        })
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update({
            # Add module dependencies
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
            "gaze_offset_data_loader.data_path": "./io/data/eval_data_final.csv",
        })
        return args

    @torch.no_grad()
    def run(self):
        df = self.gaze_offset_data_loader.generate_df()
        self.gaze_offset_model.load(self.opt.model_fn)

        def analyze_df(df):
            data_hst = self.gaze_offset_data_loader.generate_histogram(df, self.opt.trange, self.opt.nbins)

            input, _ = self.gaze_offset_data_loader.generate_tensors(df, self.opt.feature_columns)
            model_hst = self.gaze_offset_model.generate_histogram(input, self.opt.trange, self.opt.nbins)

            kldiv = rel_entr(data_hst, model_hst).sum()
            return kldiv

        def analyze_split(split_name):
            kldiv_dict = {}
            for id in df[split_name].unique():
                kldiv_dict[id] = analyze_df(df[df[split_name] == id])
            kldiv_df = pd.DataFrame.from_dict(kldiv_dict, orient="index")
            return kldiv_df

        col_names = ["c_id", "scene_id", "u_id"]
        conditions = product(*[df[col].unique().tolist() for col in col_names])
        kldiv_out = []
        for condition in conditions:
            subset_df = df
            for i, col in enumerate(col_names):
                subset_df = subset_df[subset_df[col] == condition[i]]
            kldiv = analyze_df(subset_df)
            kldiv_out.append([*condition, kldiv])

        out_df = pd.DataFrame(kldiv_out, columns=[*col_names, "kldiv"])
        out_df.to_csv(self.opt.output_fn, index=False)


if __name__ == "__main__":
    JointModelDatasplitAnalysis().initialize().run()
