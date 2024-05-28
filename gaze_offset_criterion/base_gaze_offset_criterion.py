from invoker import Module


class BaseGazeOffsetCriterion(Module):
    @classmethod
    def args(cls):
        # Specify arguments to pass from command line
        return {}
