import torch
import torch.nn as nn

from .base_gaze_offset_model import BaseGazeOffsetModel

class RbfGazeOffsetModel(BaseGazeOffsetModel):
    @classmethod
    def args(cls):
        args = super().args()
        args.update({
            "num_centres": 4,
            "rng_seed": 2,
        })
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.centres = nn.Parameter(
            torch.Tensor(self.opt.out_dims, self.opt.num_centres, self.opt.in_dims))
        self.scale = nn.Parameter(torch.Tensor(self.opt.out_dims))
        self.weights = nn.Parameter(torch.Tensor(self.opt.out_dims, self.opt.num_centres))
        self.reset_parameters()

    def reset_parameters(self):
        torch.manual_seed(self.opt.rng_seed)
        nn.init.normal_(self.centres, 0, 1)
        nn.init.normal_(self.weights, 0, 1)
        nn.init.constant_(self.scale, 1)

    def forward(self, input):
        squared_coordinate_differences = (input[:, None, None, :] - self.centres[None, ...]) ** 2
        squared_distances = squared_coordinate_differences.sum(dim=-1)
        gaussian_distances = torch.exp(-squared_distances / (2 * self.scale[None, :, None]**2))
        output = (gaussian_distances * self.weights[None, ...]).sum(dim=-1)
        return output
