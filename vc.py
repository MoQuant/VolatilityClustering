stocks = ['SPY','AAPL','GOOGL','NFLX','AMZN',
           'META','WFC','GS','JPM','NVDA','ORCL',
           'PLTR','MSFT','AMZN','F','TSLA','JNJ',
           'HD','WMT','PLTR','BLK','VMW','APO','NOC',
           'RTX','BA','LMT']

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

colors = lambda c: 'red' if c == 0 else 'orange' if c == 1 else 'green' if c == 2 else 'blue' if c == 3 else 'purple'

def Ellipses(data, color):
    U = np.array(data[color])
    x, y = U[:, 0], U[:, 1]
    a = np.arange(0, 2*np.pi+np.pi/16, np.pi/16)
    mux = np.mean(x)
    muy = np.mean(y)
    sdx = np.std(x)
    sdy = np.std(y)
    ex, ey = [], []
    for t in a:
        ui = mux + sdx*np.cos(t)
        uj = muy + sdy*np.sin(t)
        ex.append(ui)
        ey.append(uj)
    return ex, ey

def StockData():
    close = []
    t0 = 200
    for stock in stocks:
        df = pd.read_csv(f'{stock}.csv')
        close.append(df['adjClose'].values.tolist()[-t0:])
    return np.array(close).T

def RollingVol(R):
    window = 30
    result = []
    for i in range(window, len(R)):
        x = R[i-window:i]
        m, n = x.shape
        mu = (1/m)*np.ones(m).dot(x)
        cv = (1/(m-1))*(x - mu).T.dot(x - mu)
        sd = np.sqrt(np.diag(cv)).tolist()
        result.append(sd)
    return np.array(result)

C = StockData()
R = C[1:]/C[:-1] - 1.0

RV = RollingVol(R)

scaler = StandardScaler()
vol = scaler.fit_transform(RV.T)

clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
cluster_labels = kmeans.fit_predict(vol)

pca = PCA(n_components=2)
vpca = pca.fit_transform(vol)

fig = plt.figure()
ax = fig.add_subplot(111)

plt_colors = list(map(colors, cluster_labels))

data = {'red':[],'orange':[],'green':[],'blue':[],'purple':[]}

for stock, xcolor, (xx, yy) in zip(stocks, plt_colors, vpca):
    data[xcolor].append([xx, yy])
    ax.scatter(xx, yy, color=xcolor)
    ax.annotate(stock, xy=(xx, yy))

for xcolor in plt_colors:
    ex, ey = Ellipses(data, xcolor)
    ax.plot(ex, ey, color=xcolor)

ax.set_title("Stocks Grouped by Volatility Behavior")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.show()