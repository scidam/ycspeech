from matplotlib.mlab import griddata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_excel("data.xlsx")

minx, maxx = data['Y'].min(), data['Y'].max()
miny, maxy = data['X'].min(), data['X'].max()


xi = np.linspace(minx,maxx, 100)
yi = np.linspace(miny, maxy, 200)
# grid the data.
zi = griddata(data['Y'], data['X'], data['VISOT'], xi, yi, interp='linear')

plt.contourf(xi, yi, zi)
plt.show()
