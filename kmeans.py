import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from numpy import genfromtxt

df1=pd.read_csv("wine_train.csv")
df1["content"]=df1["free sulfur dioxide"]/df1["total sulfur dioxide"]+df1["alcohol"]+df1["pH"]-df1["fixed acidity"]-df1["volatile acidity"]-df1["citric acid"]
df36=df1[['density','content']]
df3=df36.values
colors = 10*["g","r","c","b","k"]
class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            for c in self.centroids:
                color = colors[c]
                for f in self.classifications[c]:
                    plt.scatter(f[0],f[1],marker="x",color=color,s=150,linewidth=4)
            for c in self.centroids:
                plt.scatter(self.centroids[c][0],self.centroids[c][1],marker="o",color="b",s=150,linewidth=5)
            plt.show()
            if optimized:
                break
				
clf = K_Means()
clf.fit(df3)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1],
                marker="o", color="k", s=150, linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color=color, s=150, linewidths=5)

plt.show()