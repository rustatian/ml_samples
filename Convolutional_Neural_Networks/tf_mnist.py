import numpy as np
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


train = pd.read_csv('train.csv')
print(train.head())
print("Shape is: {sh}".format(sh=train.shape))

f, ax = plt.subplots(figsize=(7, 6))

# plt.title('Correlation plot of a 100 columns in the MNIST dataset')
# Draw the heatmap using seaborn
# sns.heatmap(train.ix[:,100:300].astype(float).corr(),linewidths=0, square=True, cmap="viridis", xticklabels=False,
# yticklabels= False, annot=True)
# plt.show()



