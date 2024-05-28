import matplotlib.pyplot as plt


def configure_mpl():
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
            \usepackage{bm}
            \usepackage{textcomp}
        """,
        "font.size": 16,
    })
    ### PATCH START ###
    # https://stackoverflow.com/questions/16488182/removing-axes-margins-in-3d-plot
    from mpl_toolkits.mplot3d.axis3d import Axis
    if not hasattr(Axis, "_get_coord_info_old"):
        def _get_coord_info_new(self, renderer):
            mins, maxs, centers, deltas, tc, highs = self._get_coord_info_old(renderer)
            mins += deltas / 4
            maxs -= deltas / 4
            return mins, maxs, centers, deltas, tc, highs
        Axis._get_coord_info_old = Axis._get_coord_info  
        Axis._get_coord_info = _get_coord_info_new
    ### PATCH END ###
