#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from scipy.stats import kruskal
import torch

from invoker import Script
from util.mpl import configure_mpl


class JointModelAblationAnalysis(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_suffix": "_rbf_gaze_offset_model_pilot_data_final.pth",
            "nbins": 50,
            "trange": [0, 1200],
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
            "gaze_offset_model.in_dims": len(feature_columns),
        })
        return args

    @torch.no_grad()
    def run(self):
        df = self.gaze_offset_data_loader.generate_df()

        logging.info(
            kruskal(
                df[df.saccade_ang == 0].offset_time,
                df[df.vergence_ang == 0].offset_time,
                df[~(df.saccade_ang == 0) & ~(df.vergence_ang ==0)].offset_time
            )
        )

        def analyze_df(uid, df):
            cid_to_kldiv = {}
            for cid in sorted(df.c_id.unique()):
                cond_df = df[df.c_id == cid]
                data_hst = self.gaze_offset_data_loader.generate_histogram(cond_df, self.opt.trange, self.opt.nbins)

                self.gaze_offset_model.load(f"{self.opt.model_label}{self.opt.model_suffix}")
                input, _ = self.gaze_offset_data_loader.generate_tensors(cond_df, self.opt.feature_columns)
                model_hst = self.gaze_offset_model.generate_histogram(input, self.opt.trange, self.opt.nbins)

                model_hst += 1e-12  # Avoid infinity division for KLDiv

                kldiv = rel_entr(data_hst, model_hst).sum()
                cid_to_kldiv[cid] = kldiv
            logging.debug("KLDiv of %s on %s uid= '%s': %s",
                self.gaze_offset_data_loader.data_path.stem, self.opt.model_label, uid,
                " ".join([f"cid{int(cid)}: {kldiv:.3f}" for cid, kldiv in cid_to_kldiv.items()]),
            )
            logging.info("Average of uid= '%s': %.3f", uid, np.array([*cid_to_kldiv.values()]).mean())

        uids = df["u_id"].unique()
        # Analyze aggregate
        analyze_df("ALL", df)

        # Analyze per person
        split_dfs = {uid: df[df["u_id"] == uid] for uid in uids}
        for uid, split_df in split_dfs.items():
            analyze_df(uid, split_df)
        logging.debug("nbins = %d, time_range = (%.3f, %.3f)", self.opt.nbins, *self.opt.trange)


if __name__ == "__main__":
    configure_mpl()
    JointModelAblationAnalysis().initialize().run()
