from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def model_init():
    regressor = GradientBoostingRegressor(
        loss='ls',
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.9,
        criterion='friedman_mse',  # The function to measure the quality of a split
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        #                                 min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=None,
        max_features=None,
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        presort='auto')
    gbr = GridSearchCV(regressor,
                       cv=5,
                       param_grid={"n_estimators": [50, 100, 150],
                                   "learning_rate": [1e0, 0.1, 1e-2],
                                   "subsample": [0.1, 0.5, 0.9]},
                       scoring='neg_mean_absolute_error',
                       return_train_score=True,
                       refit=True,
                       iid=True)
    return gbr


def gbr_reward(features, properties):
    gbr = model_init()
    gbr.fit(features, properties)
    return gbr.best_score_
