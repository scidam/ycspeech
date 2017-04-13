import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd


# Loading data from file
data = pd.read_excel("data.xlsx")

# Getting drawing box
minx, maxx = data['Y'].min(), data['Y'].max()
miny, maxy = data['X'].min(), data['X'].max()

# regular grid
xi = np.linspace(minx,maxx, 100)
yi = np.linspace(miny, maxy, 200)


def plot_map(data, colname='DOM1', cmap='jet', size=10):
    _cmap = plt.get_cmap(cmap, len(data[colname].unique()))
    result_df = pd.DataFrame()
    colors = []
    for sp in data[colname].unique():
        cdata = data[getattr(data, colname) == sp]
        colors.extend([sp] * len(cdata))
        result_df = result_df.append(cdata)
    result_df = result_df.join(pd.DataFrame({'C': colors}))
    res = plt.scatter(result_df['Y'], result_df['X'], c=result_df['C'],
                      cmap=_cmap, s=size)
    plt.gca().set_title(colname)
    return res




res = plot_map(data, colname='DOM1', size=60)
plt.colorbar(res)





plt.show()

