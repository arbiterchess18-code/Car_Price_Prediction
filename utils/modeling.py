import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_regression_model(name: str):
    if name == 'Linear Regression':
        return LinearRegression()
    if name == 'Decision Tree Regressor':
        return DecisionTreeRegressor(random_state=42)
    if name == 'Random Forest Regressor':
        return RandomForestRegressor(random_state=42)
    if name == 'KNN Regressor':
        return KNeighborsRegressor()
    if name == 'XGBoost Regressor':
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(random_state=42, verbosity=0)
        except ImportError:
            raise ImportError('XGBoost is not installed. Install xgboost in your environment.')
    raise ValueError(f'Model {name} is not supported.')


def get_hyperparameter_grid(name: str):
    if name == 'Linear Regression':
        return None
    if name == 'Decision Tree Regressor':
        return {
            'model__max_depth': [None, 5, 10, 15],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4],
        }
    if name == 'Random Forest Regressor':
        return {
            'model__n_estimators': [100, 150, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5],
        }
    if name == 'KNN Regressor':
        return {
            'model__n_neighbors': [3, 5, 7, 9],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2],
        }
    if name == 'XGBoost Regressor':
        return {
            'model__n_estimators': [100, 150],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2],
        }
    return None


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
    }


def train_model_pipeline(
    X,
    y,
    preprocessor,
    model,
    param_grid=None,
    search_type: str = None,
    split_ratio: float = 0.2,
    random_state: int = 42,
    use_cross_validation: bool = False,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_ratio, random_state=random_state
    )
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    best_params = None

    if param_grid is not None and search_type:
        if search_type == 'GridSearchCV':
            searcher = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='r2')
        else:
            searcher = RandomizedSearchCV(
                pipeline,
                param_grid,
                n_iter=10,
                cv=3,
                random_state=random_state,
                n_jobs=-1,
                scoring='r2',
            )
        searcher.fit(X_train, y_train)
        pipeline = searcher.best_estimator_
        best_params = searcher.best_params_
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    metrics['best_params'] = best_params
    metrics['y_test'] = y_test
    metrics['y_pred'] = y_pred

    if use_cross_validation:
        cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()

    return pipeline, metrics
