from .base_gaze_offset_criterion import BaseGazeOffsetCriterion


class MseGazeOffsetCriterion(BaseGazeOffsetCriterion):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {}

    def loss(self, output, label, debug=False):
        return ((output - label) ** 2).mean(axis=0)
