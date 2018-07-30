import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

def plot_format():

    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    rc('font', size=SMALL_SIZE, **{'family': 'sans-serif', 'sans-serif': ['sans-serif']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams.update({'errorbar.capsize': 2})

