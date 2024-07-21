import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_binary_data(X: np.array, y: np.array):
    df = pd.DataFrame(X, columns=["var1", "var2"])
    df["y"] = y
    df["y"] = pd.factorize(y)[0]
    sns.scatterplot(data=df, x="var1", y="var2", hue="y")
    plt.show()
