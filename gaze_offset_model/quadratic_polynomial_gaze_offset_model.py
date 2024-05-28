import torch
import torch.nn as nn

from .base_gaze_offset_model import BaseGazeOffsetModel

class QuadraticPolynomialGazeOffsetModel(BaseGazeOffsetModel):
    @classmethod
    def args(cls):
        args = super().args()
        args.update({
            "rng_seed": 1,
        })
        return args

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        CONIC_SECTION_COEFFICIENT = 6
        torch.manual_seed(self.opt.rng_seed)
        self.weights = nn.Linear(CONIC_SECTION_COEFFICIENT, self.opt.out_dims, bias=False)
        self.activation = nn.ELU()

    def forward(self, input):
        output = self._polynomial_features(input)
        output = self.weights(output)
        ELU_SHIFT = 1 + 1e-9  # add epsilon for numerical stability
        output = self.activation(output) + ELU_SHIFT
        return output

    def _polynomial_features(self, input):
        xx = input[:,0] ** 2
        yy = input[:,1] ** 2
        xy = input[:,0] * input[:,1]
        x = input[:,0]
        y = input[:,1]
        c = torch.ones_like(input[:,0])
        features = torch.stack([xx, yy, xy, x, y, c], dim=-1)
        return features
