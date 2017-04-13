import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
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


# ----------- Global classifier comparison ------------
names = ["Nearest Neighbors", "Linear SVM",
                  "Decision Tree", "Random Forest",  "AdaBoost",
                  "Naive Bayes", "LDA"]

classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        DecisionTreeClassifier(max_depth=4),
        RandomForestClassifier(max_depth=4, n_estimators=10, max_features=1),
        AdaBoostClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis()]

#---------------------------------------------------------
data = data.dropna()
y = np.array(data['DOM1'].values.tolist())
X = np.array(data[['SKLON', 'SK_KRYT', 'TIP_POCH', 'GIDRO', 'TEMP', 'OSADK', 'VISOT']].values.tolist())
for name, clf in zip(names, classifiers):
   res = cross_val_score(clf,X, y, cv=10)
   print('Classiification results for %s, is %s' % (name, np.mean(res)))




import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
print('RESULT AUTO:', cross_val_score(cls, X,  y))

