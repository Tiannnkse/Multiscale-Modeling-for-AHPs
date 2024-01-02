# author： Tian_Gui_L
# datetime： 2023/12/30 16:10
# ide： PyCharm

import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor as RT
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


def black_box_function(n_estimators, min_samples_split, min_samples_leaf, max_features, max_depth, random_state):
    model = RT(n_estimators=int(n_estimators),
               min_samples_split=int(min_samples_split),
               min_samples_leaf=int(min_samples_leaf),
               max_features=min(max_features, 0.999),  # float
               max_depth=int(max_depth), random_state=int(random_state)
               )
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test)

    # Calculate R2 score as the objective function
    r2 = r2_score(y_test, y_pred)

    return r2


if __name__ == '__main__':
    #   Predict SCP values with Adsorption Performance
    df = pd.read_csv("../database_COP.csv")

    labels = np.array(df['COP_c']).reshape(-1)
    features = df.drop(df[['REFCODE', 'COP_c']], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.8, random_state=1)


    # Define the parameter space
    param_space = {
        'n_estimators': (10, 500),
         'min_samples_split': (2, 50),
         'min_samples_leaf': (1, 50),
         'max_features': (0.01, 0.999),
         'max_depth': (5,50),
         'random_state':(1,100)
    }
    # Get the best parameters
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=param_space,
    )

    optimizer.maximize(init_points=5, n_iter=1000)
    best_params = optimizer.max['params']

    # best_params with Adsorption Performance
    # best_params = {'max_depth': 18.901796818770396,
    #                'max_features': 0.999,
    #                'min_samples_leaf': 5.054746524499222,
    #                'min_samples_split': 12.960885740814126,
    #                'n_estimators': 96.02313291894026,
    #                'random_state': 8.740085037828312
    #                }

    # Create the final GBR model using the best parameters
    final_model = RT(
        max_features=best_params['max_features'],
        n_estimators=int(best_params['n_estimators']),
        max_depth=int(best_params['max_depth']),
        min_samples_leaf=int(best_params['min_samples_leaf']),
        min_samples_split=int(best_params['min_samples_split']),
        random_state=int(best_params['random_state'])
    )

    final_model.fit(X_train, y_train)

    r2 = final_model.score(X_test, y_test)
    r2_train = final_model.score(X_train, y_train)

    train_predictions = final_model.predict(X_train)
    test_predictions = final_model.predict(X_test)

    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))

    print(f'R2 train:{r2_train}, R2 test: {r2}')
