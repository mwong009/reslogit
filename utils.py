import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm, colors

def hinton(matrix, tstat=None, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    matrix = matrix - np.mean(matrix, axis=1, keepdims=True)

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max(axis=0)) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):

        size = np.sqrt(np.abs(w) / max_weight[y])
        color = 'white' if w > 0 else 'black'

        if tstat is not None:
            if tstat[x, y] >= 1:
                edge = 'red'
            else:
                edge = color
        else:
            edge = color

        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=edge)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def get_corr_visu(mat, figsize=(6, 6), title=''):
    '''
    Parameters
    ----------
    Args:
        mat(pandas.DataFrame): correlation matrix
    '''
    assert type(mat) == pd.core.frame.DataFrame
    assert len(mat) == len(mat.columns)

    choices = list(mat.columns)

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels(
        ['']+choices,
        rotation='45',
        va='bottom',
        ha="left",
        rotation_mode="anchor"
    )
    ax.set_yticklabels(['']+choices)
    ax.text(3, 7, title, ha='center', va='bottom', size=13)

    # define a colormap
    colormap = cm.get_cmap('coolwarm_r')

    ax.matshow(mat, alpha=0.6, cmap=colormap)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.grid(which='minor', ls=' ')

    # Add the text
    size = len(choices)
    start, end = 0, size
    skip = (end - start) / size
    x_pos = np.linspace(start=start-1, stop=end-1, num=size, endpoint=False)
    y_pos = np.linspace(start=start-1, stop=end-1, num=size, endpoint=False)

    for y_idx, y in enumerate(y_pos):
        for x_idx, x in enumerate(x_pos):
            label = mat.iloc[y_idx, x_idx].round(2)
            ax.text(
                x=x + skip, y=y + skip, s=label,
                color='black', ha='center', va='center', weight='bold', size=12
            )
