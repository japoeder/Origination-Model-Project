import plotly.express as px
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd

def plot_imbalance(data='df', target='target'):
#PCA is performed for visualization only

    pca= PCA(n_components=2)
    creditcard_2d= pd.DataFrame(pca.fit_transform(data.drop(target, axis=1)))
    creditcard_2d= pd.concat([creditcard_2d, data['Class']], axis=1)
    creditcard_2d.columns= ['x', 'y', 'Class']
   
    return sns.lmplot(x='x', y='y', data=creditcard_2d, fit_reg=False, hue='Class')