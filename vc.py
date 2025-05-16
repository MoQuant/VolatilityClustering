# Selected stock tickers
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

# Gives each volatility cluster its own color
colors = lambda c: 'red' if c == 0 else 'orange' if c == 1 else 'green' if c == 2 else 'blue' if c == 3 else 'purple'

# Generates an ellipse around a set of datapoints to signify each class they are grouped in
def Ellipses(data, color):
    # Pulls out each volatility group
    U = np.array(data[color])

    # Builds x and y 
    x, y = U[:, 0], U[:, 1]

    # Ellipse Range
    a = np.arange(0, 2*np.pi+np.pi/16, np.pi/16)
    mux = np.mean(x)
    muy = np.mean(y)
    sdx = np.std(x)
    sdy = np.std(y)
    ex, ey = [], []
    for t in a:
        # Uses means as center point and expands ellipse to standard deviations of the x and y features
        # and appends them to lists which will be plotted
        ui = mux + sdx*np.cos(t)
        uj = muy + sdy*np.sin(t)
        ex.append(ui)
        ey.append(uj)
    return ex, ey

# Extracts the past 200 days of stock data to conduct analysis on
def StockData():
    close = []
    t0 = 200
    for stock in stocks:
        df = pd.read_csv(f'{stock}.csv')
        close.append(df['adjClose'].values.tolist()[-t0:])
    return np.array(close).T

# Calculates the rolling volatility which is then analyzed
def RollingVol(R):
    window = 30
    result = []
    for i in range(window, len(R)):
        x = R[i-window:i]
        m, n = x.shape
               
        # Captures all stocks standard deviations (volatility)
        mu = (1/m)*np.ones(m).dot(x)
        cv = (1/(m-1))*(x - mu).T.dot(x - mu)
        sd = np.sqrt(np.diag(cv)).tolist()
        result.append(sd)
               
    return np.array(result)

# Load close prices
C = StockData()

# Calculate rate of returns
R = C[1:]/C[:-1] - 1.0

# Calculate rolling volatility
RV = RollingVol(R)

# Transform Rolling Volatility data to be passed to the KMeans function
scaler = StandardScaler()
vol = scaler.fit_transform(RV.T)

# Set a number of clusters equal to 5 and generate the clusters with the input vol (which is derived from Rolling Vol) 
# in order to generate the cluster labels
clusters = 5
kmeans = KMeans(n_clusters=clusters, random_state=0, n_init='auto')
cluster_labels = kmeans.fit_predict(vol)

# Utilize principal component analysis to reduce the dimensions of the dataset to just 2 components (2D graph)
pca = PCA(n_components=2)

# Transform vol into the two principal components
vpca = pca.fit_transform(vol)

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111)

# Extract colors for cluster labels
plt_colors = list(map(colors, cluster_labels))

# Stores volatilities based on their group
data = {'red':[],'orange':[],'green':[],'blue':[],'purple':[]}

# Takes a list of stocks, colors, and principal components and graphs stocks in 
# their various groups
for stock, xcolor, (xx, yy) in zip(stocks, plt_colors, vpca):
    data[xcolor].append([xx, yy])
    ax.scatter(xx, yy, color=xcolor)
    ax.annotate(stock, xy=(xx, yy))

# Plots the ellipses on top of the clusters to show which stocks deviate away from their group
for xcolor in plt_colors:
    ex, ey = Ellipses(data, xcolor)
    ax.plot(ex, ey, color=xcolor)

ax.set_title("Stocks Grouped by Volatility Behavior")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
plt.show()
