import numpy as np


def plot_history(ax, episode_count: int, history: dict, label:str):
    """ Constructs a matplotlib.ax with the information of a history dictionary

    Arguments:
        ax (AxesSubplot): axes where to plot the information
        episode_count (int): array in the X axis
        history (dict): container of training history
        label (str): label to add in ax legend
    
    Returns:
        AxesSubplot
    """

    tensor_to_list = lambda t: np.array(t).tolist()
    x = np.arange(episode_count)

    for k in history.keys():
        
        y = history[k] if type(history[k]) == list else tensor_to_list(history[k])
        ax.plot(x, y, label=f"{k} {label}")

    ax.set_xlabel('Number of Episodes', fontsize = 14)
    ax.set_ylabel(label.capitalize(),   fontsize = 14)
    ax.legend(prop={'size': 12})

    return ax


# ---------------------------------------------------------------------------------
# end