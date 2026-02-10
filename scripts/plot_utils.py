import matplotlib.pyplot as plt
import numpy as np

def minmaxscale(signal):
    smin = signal.min(axis=0)
    smax = signal.max(axis=0)
    signal = (signal - smin)/(smax - smin)
    return signal

def plot_bold(bold):
    bold = np.array(bold)
    bold = minmaxscale(bold)
    plt.figure(figsize=(6,12))
    plt.plot(range(bold.shape[1])+3*bold, linewidth=0.5)
    plt.show()

    
def basic_3d_sweep_plot(sweep_df, p1_name, 
    p2_name, p3_name, var_to_plot):
    # Use interactive notebook backend
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    #%matplotlib widget

    # Extract coordinates and variable to plot
    x = sweep_df[p1_name].astype(float)
    y = sweep_df[p2_name].astype(float)
    z = sweep_df[p3_name].astype(float)
    c = sweep_df[var_to_plot].astype(float)

    sizes = 1 + (5*(c - c.min()) / (c.max() - c.min()))**4

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=sizes, alpha=0.4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(var_to_plot)

    ax.set_xlabel(p1_name)
    ax.set_ylabel(p2_name)
    ax.set_zlabel(p3_name)

    plt.title(f'3D Scatter of {var_to_plot}')
    plt.tight_layout()
    plt.show()