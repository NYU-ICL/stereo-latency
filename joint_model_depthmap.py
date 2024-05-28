#!/usr/bin/env python
import logging
from pathlib import Path

from skimage import filters
import imageio.v3 as iio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch

from invoker import Script
from util import exrio


class JointModelDepthmap(Script):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        args = super().args()
        args.update({
            "model_label": "full",
            "model_fn": "full_rbf_gaze_offset_model_pilot_data_final.pth",
            "image_path": "./io/data/heatmap_analysis_img.exr",
            "images_root": "./io/images",
            "hfov": 24,
            "vfov": 24,
            "ipd": 0.065,  # in m
            "gaze_src_idx": np.array([640, 640]),
            "display": False,
        })
        return args

    @classmethod
    def modules(cls):
        mods = super().modules()
        mods.update({
            # Add module dependencies
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
        })
        return args

    @torch.no_grad()
    def run(self):
        self.gaze_offset_model.load(self.opt.model_fn)

        def depth2vang(depth):
            RAD2DEG = 180 / np.pi
            return np.arctan(self.opt.ipd / 2 / depth) * 2 * RAD2DEG

        def pixel2sang(pix):
            return self.opt.hfov / depth_map.shape[1] * pix

        def process_img(depth_map, src_idx):
            # Compute src angles
            src_vang = depth2vang(depth_map[*src_idx, 0])
            src_sang = pixel2sang(src_idx)

            # Compute movement amplitudes
            vamp_map = np.abs(depth2vang(depth_map[...,0]) - src_vang)
            du = np.linspace(0, self.opt.hfov, rgb_img.shape[1]) - src_sang[1]
            dv = np.linspace(0, self.opt.vfov, rgb_img.shape[0]) - src_sang[0]
            duu, dvv = np.meshgrid(du, dv)
            samp_map = (duu ** 2 + dvv ** 2) ** 0.5

            # Apply model estimation
            amp_map = np.stack([vamp_map, samp_map], axis=-1)
            amp_vec = torch.tensor(amp_map.reshape(-1, amp_map.shape[-1]))
            mean, std, decay = self.gaze_offset_model(amp_vec).numpy().T
            mean = mean.reshape(vamp_map.shape)
            std = std.reshape(vamp_map.shape)
            decay = decay.reshape(vamp_map.shape)

            offset_mean_map = mean + decay
            offset_std_map = (std**2 + decay**2) ** 0.5

            # Compute ignore values masks
            ignore_vamp_bin_mask = (vamp_map < -8.4) | (vamp_map > 8.4)
            inside_samp_bin_mask = (samp_map < 4)
            outside_samp_bin_mask = (samp_map > 12)
            ignore_all_bin_mask = ignore_vamp_bin_mask | inside_samp_bin_mask | outside_samp_bin_mask

            return vamp_map, offset_mean_map, offset_std_map, ignore_all_bin_mask

        def hack_depthmap(depth_map, target_depth):
            output_map = depth_map.copy()
            top_half = output_map[:700,...]
            top_mask = top_half < 2.14
            bottom_mask = np.zeros_like(output_map[700:,...], dtype=bool)
            mask = np.concatenate([top_mask, bottom_mask], axis=0)
            output_map[mask] = target_depth
            return output_map

        def bin2alpha(bin_mask):
            return filters.gaussian(bin_mask, sigma=3)

        rgb_img, depth_map = exrio.imread(Path(self.opt.image_path))

        depths = [1.0, 7, 25]
        depth_maps = []
        vamp_maps = []
        offset_mean_maps = []

        valid_offsets = []
        for depth in depths:
            # Compute prediction
            new_depth_map = hack_depthmap(depth_map, depth)
            vamp_map, offset_mean_map, _, ignore_bin_mask = process_img(new_depth_map, self.opt.gaze_src_idx)

            valid_offsets.append(offset_mean_map[~ignore_bin_mask])

            depth_maps.append(new_depth_map[...,0])
            vamp_maps.append(vamp_map)
            offset_mean_maps.append(offset_mean_map)

        for o in offset_mean_maps:
            print(o[260,300], o[315,715], o[400,1100], o[1100,400])

        valid_offsets = np.concatenate(valid_offsets)
        logging.info("Depth Range: [%.3f, %.3f]", np.stack(depth_maps).min(), np.stack(depth_maps).max())
        logging.info("Vamp Range: [%.3f, %.3f]", np.stack(vamp_maps).min(), np.stack(vamp_maps).max())
        logging.info("Offset Range: [%.3f, %.3f]", valid_offsets.min(), valid_offsets.max())

        def gen_scalar_mappable(vmin, vmax, cmap):
            scalar_mappable = cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=cmap,
            )
            return scalar_mappable

        def apply_colormap(maps, mappable, apply_mask=False):
            ignore_alpha_mask = bin2alpha(ignore_bin_mask)
            imgs = []
            for m in maps:
                img = mappable.to_rgba(m)
                if apply_mask:
                    img[..., -1] *= 1 - ignore_alpha_mask
                imgs.append(img)
            return imgs

        depth_range = [1, 45]
        vamp_range = [0, 3.6]
        offset_range = [330, 450]

        depth_mappable  = gen_scalar_mappable(*depth_range, cm.viridis)
        vamp_mappable   = gen_scalar_mappable(*vamp_range, cm.viridis)
        offset_mappable = gen_scalar_mappable(*offset_range, cm.plasma)

        depth_imgs = apply_colormap(depth_maps, depth_mappable)
        vamp_imgs = apply_colormap(vamp_maps, vamp_mappable)
        offset_mean_imgs = apply_colormap(offset_mean_maps, offset_mappable, True)

        def create_colorbar(mappable, vrange):
            colorbar = np.tile(np.linspace(*vrange, 1280), (100, 1)).T
            colorbar_imgs = apply_colormap([colorbar], mappable)
            return colorbar_imgs[0]

        depth_colorbar_img = create_colorbar(depth_mappable, depth_range)
        vamp_colorbar_img = create_colorbar(vamp_mappable, vamp_range)
        offset_colorbar_img = create_colorbar(offset_mappable, offset_range)

        def plot(depth_img, vamp_img, offset_mean_img):
            fig, ax = plt.subplots()
            ax.imshow(depth_img)

            #fig, ax = plt.subplots()
            #ax.imshow(vamp_img)

            fig, ax = plt.subplots()
            ax.imshow(offset_mean_img)

        images_root = Path(self.opt.images_root)
        iio.imwrite(images_root / "image.png", (rgb_img * 255).astype(np.uint8))
        iio.imwrite(images_root / "depth_colobar.png", (depth_colorbar_img * 255).astype(np.uint8))
        iio.imwrite(images_root / "vamp_colobar.png", (vamp_colorbar_img * 255).astype(np.uint8))
        iio.imwrite(images_root / "offset_colobar.png", (offset_colorbar_img * 255).astype(np.uint8))
        for i, (d, v, o) in enumerate(zip(depth_imgs, vamp_imgs, offset_mean_imgs)):
            iio.imwrite(images_root / f"depth_{i}.png", (d * 255).astype(np.uint8))
            iio.imwrite(images_root / f"vamp_{i}.png", (v * 255).astype(np.uint8))
            iio.imwrite(images_root / f"offset_{i}.png", (o * 255).astype(np.uint8))
            plot(d, v, o)
        if self.opt.display:
            plt.show()


if __name__ == "__main__":
    JointModelDepthmap().initialize().run()
