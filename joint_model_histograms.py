#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch

from invoker import Script
from util.mpl import configure_mpl


class JointModelHistograms(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_suffix": "_rbf_gaze_offset_model_pilot_data_final.pth",
            "figure_root": "./io/figures",
            "figure_prefix": "pilot_data_histograms",
            "trange": [0, 1200],
            "hst_bins": 25,
            "pdf_bins": 100,
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

    def plot_histogram(self, ax, df, color):
        ax.hist(df, range=self.opt.trange, bins=self.opt.hst_bins, color=color, density=True)
        ax.axvline(df.mean(), color='gray', linestyle='dashed', linewidth=3)

    @torch.no_grad()
    def plot_pdf(self, ax, cond_df):
        input, _ = self.gaze_offset_data_loader.generate_tensors(cond_df, self.opt.feature_columns)
        pdf_func = self.gaze_offset_model.pdf_func(input)
        t = np.linspace(*self.opt.trange, self.opt.pdf_bins)
        ax.plot(t, pdf_func(t), color="gray", linewidth=3)

    def run(self):
        self.gaze_offset_model.load(f"{self.opt.model_label}{self.opt.model_suffix}")
        df = self.gaze_offset_data_loader.generate_df()
        CID2AXID = [
            [2, 1], [2, 2], [2, 3], [0, 1], [0, 2], [0, 3], [1, 1], [1, 2], [1, 3], [0, 1], [0, 2],
            [0, 3], [1, 1], [1, 2], [1, 3], [2, 1], [2, 2], [2, 3], [1, 0], [0, 0], [1, 0], [2, 0],
        ]
        def plot_all_histograms(fig, axs, cids):
            for cid in cids:
                axid = CID2AXID[cid]
                ax = axs[*axid]
                color = self.gaze_offset_data_loader.get_color(cid)
                # Plot graphs
                cond_df = df[df["c_id"] == cid]
                self.plot_histogram(ax, cond_df.offset_time, color)
                self.plot_pdf(ax, cond_df.groupby(self.opt.feature_columns).mean("offset_time").reset_index())

        def configure_all_axes(fig, axs):
            for ax in axs.flatten():
                ax.tick_params(left = False, labelleft = False, labelbottom = False)
                ax.tick_params(direction='in', length=6, width=3)
                ax.set_xlim(*self.opt.trange)
                ax.set_xticks([250, 500, 750, 1000])

        def add_axis_labels(fig, axs, divergent=False):
            LABEL_FONTSIZE = 36
            X_LABEL_SHIFT = 0.7
            X_LABEL_PAD = -30
            Y_LABEL_SHIFT = 0.18

            axs[0, 0].set_title(r'$\Delta\alpha_\textit{s} = 0^{\circ}$',
                                x=X_LABEL_SHIFT, y=1.0, pad=X_LABEL_PAD, fontsize=LABEL_FONTSIZE)
            axs[0, 1].set_title(r'$\Delta\alpha_\textit{s} = 4^{\circ}$',
                                x=X_LABEL_SHIFT, y=1.0, pad=X_LABEL_PAD, fontsize=LABEL_FONTSIZE)
            axs[0, 2].set_title(r'$\Delta\alpha_\textit{s} = 8^{\circ}$',
                                x=X_LABEL_SHIFT, y=1.0, pad=X_LABEL_PAD, fontsize=LABEL_FONTSIZE)
            axs[0, 3].set_title(r'$\Delta\alpha_\textit{s} = 12^{\circ}$',
                                x=X_LABEL_SHIFT, y=1.0, pad=X_LABEL_PAD, fontsize=LABEL_FONTSIZE)

            axs[0, 0].yaxis.set_label_coords(Y_LABEL_SHIFT, 0.5)
            axs[1, 0].yaxis.set_label_coords(Y_LABEL_SHIFT, 0.5)
            axs[2, 0].yaxis.set_label_coords(Y_LABEL_SHIFT, 0.5)

            if divergent:
                ylabels = [r'-8.4', r'-4.2', r'0']
            else:
                ylabels = [r'0', r'4.2', r'8.4']
            axs[0, 0].set_ylabel(r'$\Delta\alpha_\textit{v} = ' + ylabels[0] + '^{\circ}$', fontsize=LABEL_FONTSIZE)
            axs[1, 0].set_ylabel(r'$\Delta\alpha_\textit{v} = ' + ylabels[1] + '^{\circ}$', fontsize=LABEL_FONTSIZE)
            axs[2, 0].set_ylabel(r'$\Delta\alpha_\textit{v} = ' + ylabels[2] + '^{\circ}$', fontsize=LABEL_FONTSIZE)

            if divergent:
                axs[2, 0].xaxis.set_label_coords(0.5, Y_LABEL_SHIFT+0.15)
                axs[2, 0].set_xlabel(r'\noindent offset time\\(0-1200ms)', fontsize=36)
                axs[2, 0].set_ylabel(r'density', fontsize=LABEL_FONTSIZE)

        figure_root = Path(self.opt.figure_root)
        figure_root.mkdir(parents=True, exist_ok=True)

        # Far-to-near conditions
        far_start_cids =  [ 3,  4,  5, 12, 13, 14, 15, 16, 17, 20, 21]
        fig, axs = plt.subplots(3, 4, figsize=(15, 10), tight_layout=True, sharey='all', sharex='all')
        configure_all_axes(fig, axs)
        add_axis_labels(fig, axs)
        plot_all_histograms(fig, axs, far_start_cids)

        if not self.opt.skip_write:
            if self.opt.model_label == "full":
                plt.savefig(figure_root / f"{self.opt.figure_prefix}_convergent.pdf", bbox_inches="tight")
            else:
                plt.savefig(figure_root / f"{self.opt.figure_prefix}_convergent_{self.opt.model_label}.pdf", bbox_inches="tight")

        

        # Near-to-far conditions
        near_start_cids = [ 0,  1,  2,  6,  7,  8,  9, 10, 11, 18, 19]
        fig, axs = plt.subplots(3, 4, figsize=(15, 10), tight_layout=True, sharey='all', sharex='all')
        configure_all_axes(fig, axs)
        add_axis_labels(fig, axs, divergent=True)
        plot_all_histograms(fig, axs, near_start_cids)

        if not self.opt.skip_write:
            if self.opt.model_label == "full":
                plt.savefig(figure_root / f"{self.opt.figure_prefix}_divergent.pdf", bbox_inches="tight")
            else:
                plt.savefig(figure_root / f"{self.opt.figure_prefix}_divergent_{self.opt.model_label}.pdf", bbox_inches="tight")

        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    JointModelHistograms().initialize().run()
