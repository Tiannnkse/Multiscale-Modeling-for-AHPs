# author： Tian_Gui_L
# datetime： 2023/12/30 15:41
# ide： PyCharm

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def black_box_function(learning_rate, n_estimators, max_depth, num_leaves, min_data_in_leaf, feature_fraction,
                       bagging_fraction, lambda_l1, random_state):
    model = lgbm.LGBMRegressor(learning_rate=learning_rate,
                             n_estimators=int(n_estimators),
                             max_depth=int(max_depth),
                             num_leaves=int(num_leaves),
                             min_data_in_leaf=int(min_data_in_leaf),
                             feature_fraction=feature_fraction,  # float
                             bagging_fraction=bagging_fraction,
                             lambda_l1=lambda_l1,
                             random_state=int(random_state), verbose=-1
                             )
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate R2 score as the objective function
    r2 = r2_score(y_test, y_pred)

    return r2


if __name__ == '__main__':
    # Predict COP values with Adsorption Performance
    df = pd.read_csv("../database_COP.csv")

    labels = np.array(df['COP_c']).reshape(-1)
    features = df.drop(df[['REFCODE', 'COP_c']], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=0)

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Define the parameter space
    LGB_BO = BayesianOptimization(black_box_function, {'learning_rate': (0.01, 0.3),
                                           'n_estimators': (10, 500),
                                           'max_depth': (5, 50),
                                           'num_leaves': (5, 130),
                                           'min_data_in_leaf': (5, 30),
                                           'feature_fraction': (0.7, 1.0),
                                           'bagging_fraction': (0.7, 1.0),
                                           'lambda_l1': (0, 6),
                                           'random_state': (1, 100)
                                           })

    LGB_BO.maximize(
            init_points=5,  #执行随机搜索的步数
            n_iter=2000,   #执行贝叶斯优化的步数
        )
    best_params = LGB_BO.max['params']

    # best_params with Adsorption Performance
    # best_params = {
    #     'bagging_fraction': 0.9780322240686157,
    #     'feature_fraction': 0.9923571012876686,
    #     'lambda_l1': 0.29192022002298845,
    #     'learning_rate': 0.19838933691268262,
    #     'max_depth': 11.636313604779918,
    #     'min_data_in_leaf': 12.55442326846415,
    #     'n_estimators': 437.53210567674637,
    #     'num_leaves': 66.01501491649516,
    #     'random_state': 8.809856279143414
    # }

    # Create the final GBR model using the best parameters
    final_model = lgbm.LGBMRegressor(
        learning_rate=best_params['learning_rate'],
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        num_leaves=int(best_params['num_leaves']),
        min_data_in_leaf=int(best_params['min_data_in_leaf']),
        feature_fraction=best_params['feature_fraction'],
        bagging_fraction=best_params['bagging_fraction'],
        lambda_l1=best_params['lambda_l1'],
        random_state=int(best_params['random_state']), verbose=-1
    )

    final_model.fit(X_train, y_train)

    r2 = final_model.score(X_test, y_test)
    r2_train = final_model.score(X_train, y_train)

    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))

    print(f'R2 test: {r2}, R2 train:{r2_train},  RMSE test:{rmse_test}, RMSE train: {rmse_train}')
