import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc

def plot_format():

    plt.style.use('seaborn-colorblind')

    #cb = ["#4878D0", "#EE854A", "#6ACC64", "#D65F5F", "#956CB4", "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]

    #cb = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

    cb = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133", "#56B4E9"] #-colorblind plotting

    SMALL_SIZE = 26
    MEDIUM_SIZE = 34
    BIGGER_SIZE = 36

    rc('font', size=SMALL_SIZE, **{'family': 'sans-serif', 'sans-serif': ['sans-serif']})
    ## for Palatino and other serif fonts use:
    # rc('font',**{'family':'serif','serif':['Palatino']})
    rc('text', usetex=True)
    rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams.update({'errorbar.capsize': 2})

    return cb

def plot_save(filename):
    plt.savefig(filename, dpi=300, format='pdf', bbox_inches='tight')