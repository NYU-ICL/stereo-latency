#!/usr/bin/env python
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from invoker import Script
from util.mpl import configure_mpl


class JointModelEvalHistogram(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update({
            "model_label": "full",
            "model_fn": "full_rbf_gaze_offset_model_pilot_data_final.pth",
            "figure_root": "./io/figures",
            "figure_fn": "eval_data_histogram.pdf",
            "trange": [200, 1100],
            "dbins": 25,
            "mbins": 100,
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
        configs = super().build_config(args)
        if configs["model_label"] == "vergence":
            feature_columns = ["vergence_ang"]
        elif configs["model_label"] == "saccade":
            feature_columns = ["saccade_ang"]
        else:
            feature_columns = ["vergence_ang", "saccade_ang"]
        configs.update({
            "feature_columns": feature_columns,
            "gaze_offset_data_loader.data_path": "./io/data/eval_data_final.csv",
        })
        return configs

    def run(self):
        df = self.gaze_offset_data_loader.generate_df()
        self.gaze_offset_model.load(self.opt.model_fn)

        conditions = {
            10: (r"$\mathbf{C_m}$", "#ef476f"),
            11: (r"$\mathbf{C_l}$", "#06d6a0"),
            19: (r"$\mathbf{C_s}$", "#118ab2"),
        }

        fig, ax = plt.subplots(figsize=(5, 4.5))
        markers = ["^", "o", "x"]
        for idx, (cid, (label, color)) in enumerate(conditions.items()):
            cond_df = df[df.c_id == cid]
            data_prob_density, bin_edges = np.histogram(cond_df.offset_time,
                    range=self.opt.trange, bins=self.opt.dbins, density=True)
            bin_width = bin_edges[1] - bin_edges[0]
            data_prob = data_prob_density * bin_width
            data_cum = np.cumsum(data_prob)
            ax.scatter(bin_edges[:-1], data_cum, marker=markers[idx],
                color=color, label="data: " + label, zorder=-idx, s=40,
            )
            input, _ = self.gaze_offset_data_loader.generate_tensors(
                cond_df.groupby(self.opt.feature_columns).mean("offset_time").reset_index(),
                self.opt.feature_columns,
            )
            cdf_func = self.gaze_offset_model.cdf_func(input)
            t = np.linspace(*self.opt.trange, self.opt.mbins)
            ax.plot(t, cdf_func(t), zorder=-idx,
                color=color, label="prediction: " + label, linewidth=3)

        ax.set_ylabel("cumulative probability")
        ax.set_xlabel("offset time (ms)")
        ax.set_xlim(*self.opt.trange)
        ax.set_xticks([250, 500, 750, 1000])
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0.25, 0.50, 0.75, 1.00])
        plt.tick_params(direction="in")
        plt.tick_params(axis="y", pad=-10)
        for tick in ax.yaxis.get_majorticklabels():
            tick.set_horizontalalignment("left")
        #ax.tick_params(labelleft=False, left=False)

        handles, labels = ax.get_legend_handles_labels()
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda x: x[1]))
        ax.legend(handles, labels, fontsize=12)

        if not self.opt.skip_write:
            figure_root = Path(self.opt.figure_root)
            plt.savefig(figure_root / self.opt.figure_fn, bbox_inches='tight')

        if self.opt.display:
            plt.show()



if __name__ == "__main__":
    configure_mpl()
    JointModelEvalHistogram().initialize().run()
