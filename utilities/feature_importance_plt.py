import seaborn as sns
from matplotlib import pyplot as plt

def feature_importance_plt(feature_importance, model):

    if model == 'lightgbm':
        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]
        plt.figure(figsize=(15, 12))
        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.savefig('lgbm_importances.png')

    if model == 'randforest':
        forest_importances = pd.Series(feature_importance, index=X_test.columns).sort_values(ascending=False)
        forest_importances = forest_importances[0:20]
        fig, ax = plt.subplots(figsize=(8,15))
        width=0.55
        ax.barh(np.arange(len(forest_importances)), forest_importances, width)
        ax.set_yticks(np.arange(len(forest_importances)))
        ax.set_yticklabels(forest_importances.index)
        plt.title('Feature Importance from DT')
        ax.set_ylabel('Normalized Gini Importance')