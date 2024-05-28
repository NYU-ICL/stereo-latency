#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr
import torch

from joint_model_base import JointModelBase
from util.mpl import configure_mpl


def np2rarr(arr, varname):
    return varname + " <- " + "c(" + ", ".join([str(x) for x in arr.tolist()]) + ")"


class JointModelExcludedAnalysis(JointModelBase):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_suffix": "_rbf_gaze_offset_model_pilot_data_final.pth",
            "eval_path": "./io/data/eval_data_final.csv",
            "rng_seed": 1,  # Must match random partition sample seed
            # for K.S Test
            "kscode_path": "joint_model_excluded_analysis_ks.r",
            "qrange": [0.001, 0.999],
            "qbins": 100,
            # for KL Div
            "kldiv_path": "./io/stats/joint_model_excluded_analysis_kldiv.csv",
            "trange": [0, 1200],
            "tbins": 50,
            # For Plotting
            "figure_root": "./io/figures/",
            "pilot_figure_fn": "pilot_data_qq_plot.pdf",
            "eval_figure_fn": "eval_data_qq_plot.pdf",
            "skip_write": False,
            "display": False,
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
        # Load Data
        df = self.gaze_offset_data_loader.generate_df()

        def analyze(condition_id, df):
            # Compute KL Div
            data_hst = self.gaze_offset_data_loader.generate_histogram(df, self.opt.trange, self.opt.tbins)

            input, label = self.gaze_offset_data_loader.generate_tensors(df, self.opt.feature_columns)
            model_hst = self.gaze_offset_model.generate_histogram(input, self.opt.trange, self.opt.tbins)

            kldiv = rel_entr(data_hst, model_hst).sum()

            # Compute KS Test
            data_qnt = self.gaze_offset_data_loader.generate_quantiles(df, self.opt.qrange, self.opt.qbins)

            model_qnt = self.gaze_offset_model.generate_quantiles(input, label, self.opt.qrange, self.opt.qbins)

            kscode = [
                np2rarr(data_qnt[::10], f"d_{condition_id}") + "\n",
                np2rarr(model_qnt[::10], f"m_{condition_id}") + "\n",
                f"ks.test(d_{condition_id}, m_{condition_id}, alternative='two.sided')\n",
            ]

            return kldiv, kscode, data_qnt, model_qnt

        # Process outputs
        uid_dfs = {uid: df[df["u_id"] == uid] for uid in df["u_id"].unique()}
        kldiv_dict = {}
        kscode_lines = [
            "#!/usr/bin/env Rscript",
            "library(dgof)\n\n",
            f"# KS Test Code for {self.gaze_offset_data_loader.data_path.stem}\n",
        ]
        pilot_qnt_arr = []
        eval_qnt_arr = []

        # User partition exclusions
        for uid, uid_df in uid_dfs.items():
            self.gaze_offset_model.load(f"remove_{uid}{self.opt.model_suffix}")
            kldiv, kscode, data_qnt, model_qnt = analyze(uid, uid_df)
            kldiv_dict[uid] = kldiv
            kscode_lines.append("\n")
            kscode_lines.extend(kscode)
            pilot_qnt_arr.append((data_qnt, model_qnt))

        # Random partition exclusion
        np.random.seed(self.opt.rng_seed)
        random_df = self.gaze_offset_data_loader.generate_df().sample(frac=1-0.125)
        condition_id = "random"
        self.gaze_offset_model.load(f"remove_{condition_id}{self.opt.model_suffix}")
        kldiv, kscode, data_qnt, model_qnt = analyze(condition_id, random_df)
        kldiv_dict[condition_id] = kldiv
        kscode_lines.append("\n")
        kscode_lines.extend(kscode)
        pilot_qnt_arr.append((data_qnt, model_qnt))

        # Eval data analysis
        self.gaze_offset_data_loader.opt.data_path = self.opt.eval_path
        self.gaze_offset_data_loader.load()
        df = self.gaze_offset_data_loader.generate_df()
        self.gaze_offset_model.load(f"full{self.opt.model_suffix}")
        uid_dfs = {uid: df[df["u_id"] == uid] for uid in df["u_id"].unique()}
        for uid, uid_df in uid_dfs.items():
            _, kscode, data_qnt, model_qnt = analyze(uid, uid_df)
            kscode_lines.append("\n")
            kscode_lines.extend(kscode)
            eval_qnt_arr.append((data_qnt, model_qnt))

        # Dump outputs
        kldiv_df = pd.DataFrame.from_dict(kldiv_dict, orient="index")
        kldiv_df.to_csv(Path(self.opt.kldiv_path), header=False)

        with open(Path(self.opt.kscode_path), "w") as f:
            f.writelines(kscode_lines)

        # Plot Q-Q figure
        def plot_qq(qnt_arr, scatter_config):
            fig, ax = plt.subplots(figsize=(5, 4.5))
            plt.plot(self.opt.trange, self.opt.trange, color="black", zorder=-1)
            for i, (data_qnt, model_qnt) in enumerate(qnt_arr):
                plt.scatter(data_qnt, model_qnt, s=20, label=scatter_config[i][0], marker=scatter_config[i][1])

            plt.xlim(200, 1100)
            plt.ylim(200, 1100)
            plt.tick_params(direction="in")
            plt.tick_params(axis="y", pad=-10)
            for tick in ax.yaxis.get_majorticklabels():
                tick.set_horizontalalignment("left")
            plt.xticks([250, 500, 750, 1000])
            plt.yticks([500, 750, 1000])
            plt.legend(loc="lower right", fontsize=12, ncol=3, labelspacing=0.3, handletextpad=0.4,
                columnspacing=1.5, markerscale=2)

        figure_root = Path(self.opt.figure_root)
        PILOT_SCATTER_CONFIG = [
            ("C1", "o"),
            ("C2", "v"),
            ("C3", "^"),
            ("C4", "1"),
            ("C5", "s"),
            ("C6", "*"),
            ("C7", "x"),
            ("C8", "d"),
            ("Cr", "p"),
        ]
        EVAL_SCATTER_CONFIG = [
            ("U1", "o"),
            ("U2", "v"),
            ("U3", "^"),
            ("U4", "<"),
            ("U5", ">"),
            ("U6", "1"),
            ("U7", "2"),
            ("U8", "3"),
            ("U9", "s"),
            ("U10", "*"),
            ("U11", "x"),
            ("U12", "d"),
        ]
        plot_qq(pilot_qnt_arr, PILOT_SCATTER_CONFIG)
        plt.xlabel("model-predicted test set (ms)")
        plt.ylabel("traning set (ms)")
        if not self.opt.skip_write:
            plt.savefig(figure_root / self.opt.pilot_figure_fn, bbox_inches="tight")

        plot_qq(eval_qnt_arr, EVAL_SCATTER_CONFIG)
        plt.xlabel("model-predicted offset time (ms)")
        plt.ylabel("observed offset time (ms)")
        if not self.opt.skip_write:
            plt.savefig(figure_root / self.opt.eval_figure_fn, bbox_inches="tight")

        # Show Visualization
        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    JointModelExcludedAnalysis().initialize().run()
