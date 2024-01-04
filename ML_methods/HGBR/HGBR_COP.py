# author： Tian_Gui_L
# datetime： 2023/12/30 21:35 
# ide： PyCharm

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def black_box_function(max_iter, max_leaf_nodes, max_depth, min_samples_leaf, max_bins, random_state):
    model = HGBR(
        max_iter=int(max_iter),
        max_leaf_nodes=int(max_leaf_nodes),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        max_bins=int(max_bins),
        random_state=int(random_state)
    )

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate R2 score as the objective function
    r2 = r2_score(y_test, y_pred)

    return r2


if __name__ == '__main__':

    #   Predict COP values with Adsorption Performance
    df = pd.read_csv("../database_COP.csv")

    labels = np.array(df['COP_c']).reshape(-1)
    features = df.drop(df[['REFCODE', 'COP_c']], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=0)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    pbounds = {'max_iter': (100, 2500),
               'max_leaf_nodes': (2, 50),
               'max_depth': (1, 50),
               'min_samples_leaf': (2, 50),
               'max_bins': (2, 255),
               'random_state': (1, 100)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=2000,
    )
    best_params = optimizer.max['params']
    # best_params = {
    #     'max_bins': 235.41255281421766,
    #     'max_depth': 22.21172548139879,
    #     'max_iter': 1575.3936602860856,
    #     'max_leaf_nodes': 47.36735429538802,
    #     'min_samples_leaf': 24.942882463183565,
    #     'random_state': 27.61188083789026
    # }

    final_model = HGBR(
        max_iter=int(best_params['max_iter']),
        max_leaf_nodes=int(best_params['max_leaf_nodes']),
        max_depth=int(best_params['max_depth']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        max_bins=int(best_params['max_bins']),
        random_state=int(best_params['random_state'])
    )

    final_model.fit(X_train, y_train)

    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    r2 = final_model.score(X_test, y_test)
    r2_train = final_model.score(X_train, y_train)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))

    print(f'R2 train:{r2_train}, R2 test: {r2}')