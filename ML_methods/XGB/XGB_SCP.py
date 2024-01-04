# author： Tian_Gui_L
# datetime： 2023/12/30 22:10
# ide： PyCharm

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor as XGB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def xgb_cv(max_depth, learning_rate,random_state, n_estimators, min_child_weight, subsample, colsample_bytree, reg_alpha, gamma,seed):
    model = XGB(max_depth=int(max_depth),
          learning_rate=learning_rate,
          n_estimators=int(n_estimators),
          min_child_weight=min_child_weight,
          subsample=max(min(subsample, 1), 0),
          colsample_bytree=max(min(colsample_bytree, 1), 0),
          reg_alpha=max(reg_alpha, 0), gamma=gamma, objective='reg:squarederror',
          booster='gbtree',random_state = int(random_state),
          seed=int(seed))

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate R2 score as the objective function
    r2 = r2_score(y_test, y_pred)

    return r2


if __name__ == '__main__':
    #   Predict SCP values with Adsorption Performance
    df = pd.read_csv("../database_SCP.csv")

    labels = np.array(df['SCP']).reshape(-1)
    features = df.drop(df[['REFCODE', 'SCP']], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=1)


    # Define the parameter space
    param_space = BayesianOptimization(xgb_cv, pbounds={'max_depth': (5, 50),
                                                   'learning_rate': (0.01, 0.3),
                                                   'n_estimators': (1, 500),
                                                   'min_child_weight': (0, 20),
                                                   'subsample': (0.001, 1),
                                                   'colsample_bytree': (0.01, 1),
                                                   'reg_alpha': (0.001, 20), 'random_state': (1, 100),
                                                   'gamma': (0.001, 10), 'seed': (1, 1000)})

    param_space.maximize(n_iter=5, init_points=2000)

    best_params = param_space.max['params']

    # best_params with Adsorption Performance
    # best_params = {
    #     'colsample_bytree': 0.9561156502202054,
    #     'gamma': 8.262871785471113,
    #     'learning_rate': 0.1718316743153165,
    #     'max_depth': 30.015377827799092,
    #     'min_child_weight': 3.6701129543289235,
    #     'n_estimators': 117.182056439505,
    #     'random_state': 28.362070785159922,
    #     'reg_alpha': 11.148033183732277,
    #     'seed': 264.59690333595705,
    #     'subsample': 0.8705505652893675
    # }

    # Create the final GBR model using the best parameters
    final_model = XGB(gamma=best_params['gamma'], colsample_bytree=best_params['colsample_bytree'],
                               learning_rate=best_params['learning_rate'],
                               max_depth=int(best_params['max_depth']), min_child_weight=best_params['min_child_weight'],
                               n_estimators=int(best_params['n_estimators']),
                               reg_alpha=best_params['reg_alpha'], subsample=best_params['subsample'],
                               objective='reg:squarederror',
                               booster='gbtree', random_state=int(best_params['random_state']),
                               n_jobs=4, seed=int(best_params['seed']))

    final_model.fit(X_train, y_train)

    r2 = final_model.score(X_test, y_test)
    r2_train = final_model.score(X_train, y_train)

    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))

    print(f'R2 test: {r2}, R2 train:{r2_train}')
