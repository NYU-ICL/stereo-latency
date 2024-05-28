#!/usr/bin/env python
import logging
from pathlib import Path

from matplotlib.colors import to_hex
import matplotlib.pyplot as plt
import numpy as np

from invoker import Script
from util.mpl import configure_mpl


class PilotDataAggregate(Script):
    @classmethod
    def args(cls):
        args = super().args()
        args.update({
            # Specify arguments to pass from command line
            "figure_root": "./io/figures",
            "figure_prefix": "pilot_data_aggregate_",
            "errorbars": False,
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
        })
        return mods

    @classmethod
    def build_config(cls, args):
        # Args post-processing prior to script main exec
        configs = super().build_config(args)
        configs.update({
            # Add path keyword to store output
        })
        return configs

    def run(self):
        def initialize_ax(primary=False):
            fig, ax = plt.subplots(figsize=(3.5,3))
            ax.tick_params(left=False, bottom=False)
            ax.set_xlim([-0.5, 12.5])
            ax.set_ylim([0, 1000])
            ax.set_xticks(np.linspace(0, 12, 4))
            ax.set_yticks([])
            ax.grid(True, color="lightgrey", linewidth=1.2, linestyle="--")
            ax.set_axisbelow(True)  # Ensure ax.grid is under plotting
            if primary:
                ax.set_xlabel('saccade amplitude '+ r'$\Delta\alpha_\textit{s}(^\circ)$')
                ax.set_ylabel('offset time (ms)')
                ax.set_yticks([200, 400, 600, 800])
                ax.yaxis.set_label_coords(0.1, 0.5)
                ax.xaxis.set_label_coords(0.5, 0.1)
            return fig, ax

        def plot(ax, df, marker, linestyle="-", verbose=False):
            label = r"$\Delta\alpha_\textit{v}=" + str(df.vergence_ang.unique()[0]) + r"^\circ$"
            if verbose:
                label = "vergence amp " + label
            get_color_func = self.gaze_offset_data_loader.get_color
            colors = [to_hex(get_color_func(int(cid))) for cid in df.c_id["min"]]
            ax.scatter(df.saccade_ang, df.offset_time["mean"], c=colors, s=50, marker=marker, zorder=2)

            xyc = [p for p in zip(df.saccade_ang, df.offset_time["mean"], df.offset_time["std"], colors)]
            for i in range(len(xyc)-1):
                x, y, yerr, c = [xyc[i][0], xyc[i+1][0]], [xyc[i][1], xyc[i+1][1]], [xyc[i][2], xyc[i+1][2]], xyc[i][3]
                seg_label = label if i == len(xyc)-2 else None
                ax.plot(x, y, c=c, label=seg_label, lw=3, linestyle=linestyle, zorder=1)
                if self.opt.errorbars:
                    ax.errorbar(x, y, yerr=yerr, c=c, lw=3, fmt="none", capsize=4, zorder=1)

        df = self.gaze_offset_data_loader.generate_df()
        df = df.drop("u_id", axis=1)
        df = df.groupby(["vergence_ang", "saccade_ang"]).agg(["mean", "std", "skew", "count", "min"]).reset_index()
        logging.info("Skewness of distributions = %.3f±%.3f",
            df.offset_time["skew"].mean(), df.offset_time["skew"].std())

        figure_root = Path(self.opt.figure_root)
        errorbar_label = "" if not self.opt.errorbars else "_errorbar"

        fig, ax = initialize_ax(primary=True)
        saccade_df = df[df.vergence_ang == 0][["c_id", "vergence_ang", "saccade_ang", "offset_time"]]
        plot(ax, saccade_df, marker="o", verbose=False)
        plt.legend(loc="upper right", handlelength=2.5)

        if not self.opt.skip_write:
            plt.savefig(figure_root / f"{self.opt.figure_prefix}saccade{errorbar_label}.pdf", bbox_inches="tight")

        fig, ax = initialize_ax()
        lo_diverge_df = df[df.vergence_ang == -4.2][["c_id", "vergence_ang", "saccade_ang", "offset_time"]]
        hi_diverge_df = df[df.vergence_ang == -8.4][["c_id", "vergence_ang", "saccade_ang", "offset_time"]]
        plot(ax, lo_diverge_df, marker="^", linestyle="--")
        plot(ax, hi_diverge_df, marker="o")
        plt.legend(loc="upper right", handlelength=2.5)

        if not self.opt.skip_write:
            plt.savefig(figure_root / f"{self.opt.figure_prefix}divergent{errorbar_label}.pdf", bbox_inches="tight")

        fig, ax = initialize_ax()
        lo_converge_df = df[df.vergence_ang == 4.2][["c_id", "vergence_ang", "saccade_ang", "offset_time"]]
        hi_converge_df = df[df.vergence_ang == 8.4][["c_id", "vergence_ang", "saccade_ang", "offset_time"]]
        plot(ax, lo_converge_df, marker="^", linestyle="--")
        plot(ax, hi_converge_df, marker="o")
        plt.legend(loc="upper right", handlelength=2.5)

        if not self.opt.skip_write:
            plt.savefig(figure_root / f"{self.opt.figure_prefix}convergent{errorbar_label}.pdf", bbox_inches="tight")

        if self.opt.display:
            plt.show()

                


if __name__ == "__main__":
    configure_mpl()
    PilotDataAggregate().initialize().run()
