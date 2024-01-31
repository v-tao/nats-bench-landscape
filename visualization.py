import matplotlib.pyplot as plt

def plot_fitnesses(df, fit_header, archs_i, save=False, file_name="plot.png"):
    # given a set of achitecture indices, plots fitnesses
    fits = df.loc[archs_i, fit_header].tolist()
    plt.plot(fits)
    if save:
        plt.savefig(file_name)
