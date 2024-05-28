#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import torch

from invoker import Script
from util.mpl import configure_mpl


class JointModelVisualize(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_suffix": "_rbf_gaze_offset_model_pilot_data_final.pth",
            "figures_root": "./io/figures",
            "figure_fn": "model_surface.pdf",
            "grid_sz": (80, 80),
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
        self.gaze_offset_model.load(f"{self.opt.model_label}{self.opt.model_suffix}")

        def plot_surface():
            # Model prediction
            v = np.linspace(-8.4, 8.4, self.opt.grid_sz[0])
            s = np.linspace(4, 12, self.opt.grid_sz[1])
            vv, ss = np.meshgrid(v, s)
            inp = torch.tensor(np.stack([vv.flatten(), ss.flatten()], axis=-1), dtype=torch.float32)
            out = self.gaze_offset_model(inp)
            offset_mean = out[:,0] + out[:,2]
            offset_mean = offset_mean.numpy().flatten().reshape(self.opt.grid_sz)

            # Data aggregate
            df = self.gaze_offset_data_loader.generate_df()[["saccade_ang", "vergence_ang", "offset_time"]]
            df = df[
                (df["saccade_ang"] > 0) &\
                (df["vergence_ang"] >= -8.4) &\
                (df["vergence_ang"] <= 8.4)
            ]
            df = df.groupby(["vergence_ang", "saccade_ang"]).mean().reset_index()
            data_vv, data_ss, data_offset = df["vergence_ang"], df["saccade_ang"], df["offset_time"]

            # Surface plotting
            plt.figure(figsize=(5,5))
            ax = plt.axes(projection="3d", computed_zorder=False)
            ax.view_init(40, 60)

            # Plot twice over to fix pdf anti-aliasing for matplotlib 3d surface plots
            # https://stackoverflow.com/questions/23347726/matplotlib-surface-plot-linewidth-wrong
            for _ in range(2):
                ax.plot_surface(vv, ss, offset_mean,
                    rstride=1, cstride=1, cmap=cm.plasma, linewidths=0, antialiased=False)
            ax.plot_wireframe(vv, ss, offset_mean,
                rstride=10, cstride=10, color=(1.0, 1.0, 1.0, 0.1))
            ax.scatter3D(data_vv, data_ss, data_offset,
                s=50, c=data_offset, cmap=cm.plasma, edgecolors="black", alpha=1)

            # Line plotting
            def make_line(v, s):
                vv, ss = np.meshgrid(v, s)
                inp = torch.tensor(np.stack([vv.flatten(), ss.flatten()], axis=-1), dtype=torch.float32)
                out = self.gaze_offset_model(inp)
                offset_mean = out[:,0] + out[:,2]
                offset_mean = offset_mean.numpy().flatten().reshape(vv.shape)
                min_ss = s[offset_mean.argmin(axis=0)]
                min_tt = offset_mean.min(axis=0)
                return v[:, None], min_ss[:, None], min_tt[:, None]

            def min_line():
                v = np.linspace(0, -8.4, self.opt.grid_sz[0])
                s = np.linspace(12, 4, self.opt.grid_sz[1]*10)
                return make_line(v, s)

            m_v, m_s, m_t = min_line()

            # Axes config
            ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            ax.set_xticks([-9, -6, -3, 0, 3, 6, 9])
            ax.set_yticks([4, 8, 12])
            ax.set_xlim3d(-9, 9)
            ax.set_ylim3d(4, 12)
            ax.autoscale(enable=True, axis='both', tight=False)

            if not self.opt.skip_write:
                figures_root = Path(self.opt.figures_root)
                figures_root.mkdir(parents=True, exist_ok=True)
                plt.savefig(figures_root / self.opt.figure_fn, bbox_extra_artists=(lbl,lblx, lbly))

        plot_surface()
        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    configure_mpl()
    JointModelVisualize().initialize().run()
