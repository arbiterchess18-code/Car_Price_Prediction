import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder


def correlation_heatmap(df: pd.DataFrame):
    # Create a copy to avoid modifying original data
    df_encoded = df.copy()
    
    # Encode categorical columns
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Calculate correlation on all columns (now all numeric)
    corr = df_encoded.corr()
    num_cols = len(corr.columns)
    fig_height = max(600, num_cols * 50)
    fig_width = max(800, num_cols * 50)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Correlation Heatmap',
    )
    fig.update_layout(
        margin=dict(l=150, r=40, t=60, b=150),
        height=fig_height,
        width=fig_width,
        xaxis={'side': 'bottom'},
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_histogram(df: pd.DataFrame, columns: list):
    melted = df[columns].melt(var_name='Feature', value_name='Value')
    fig = px.histogram(
        melted,
        x='Value',
        color='Feature',
        facet_col='Feature',
        facet_col_wrap=2,
        marginal='rug',
        title='Feature Distribution Histograms',
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_boxplot(df: pd.DataFrame, columns: list):
    melted = df[columns].melt(var_name='Feature', value_name='Value')
    fig = px.box(
        melted,
        x='Feature',
        y='Value',
        color='Feature',
        title='Boxplot Analysis',
        points='outliers',
    )
    fig.update_layout(showlegend=False, margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str):
    scatter_options = {
        'x': x_col,
        'y': y_col,
        'title': f'Scatter plot: {x_col} vs {y_col}',
        'labels': {x_col: x_col, y_col: y_col},
    }
    try:
        fig = px.scatter(df, trendline='ols', **scatter_options)
    except Exception:
        fig = px.scatter(df, **scatter_options)
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_pairplot(df: pd.DataFrame, columns: list):
    fig = px.scatter_matrix(
        df[columns],
        dimensions=columns,
        title='Pairplot for Selected Features',
        labels={col: col for col in columns},
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_feature_importance(model, feature_names: list):
    importance = None
    if hasattr(model, 'feature_importances_'):
        importance = np.asarray(model.feature_importances_)
    elif hasattr(model, 'coef_'):
        importance = np.abs(np.asarray(model.coef_))
    if importance is None:
        return None

    if importance.ndim > 1:
        if importance.shape[0] == 1:
            importance = importance.ravel()
        else:
            importance = importance.mean(axis=0)

    importance = importance.ravel()
    if len(feature_names) != len(importance):
        min_len = min(len(feature_names), len(importance))
        feature_names = feature_names[:min_len]
        importance = importance[:min_len]

    if len(feature_names) == 0 or len(importance) == 0:
        return None

    importance_df = pd.DataFrame(
        {'feature': feature_names, 'importance': importance}
    ).sort_values('importance', ascending=False).head(20)
    fig = px.bar(
        importance_df,
        x='importance',
        y='feature',
        orientation='h',
        title='Feature Importance',
        text='importance',
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=150, r=40, t=60, b=40))
    return fig


def plot_actual_vs_predicted(y_test, y_pred):
    chart_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig = px.scatter(
        chart_df,
        x='Actual',
        y='Predicted',
        title='Actual vs Predicted Values',
        labels={'Actual': 'Actual Price', 'Predicted': 'Predicted Price'},
    )
    fig.add_shape(
        type='line',
        x0=chart_df['Actual'].min(),
        y0=chart_df['Actual'].min(),
        x1=chart_df['Actual'].max(),
        y1=chart_df['Actual'].max(),
        line=dict(color='red', dash='dash'),
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_metric_comparison(metrics: dict):
    metric_df = pd.DataFrame(
        [
            {'Metric': 'MAE', 'Value': metrics.get('mae', 0)},
            {'Metric': 'MSE', 'Value': metrics.get('mse', 0)},
            {'Metric': 'RMSE', 'Value': metrics.get('rmse', 0)},
            {'Metric': 'R2 Score', 'Value': metrics.get('r2', 0)},
        ]
    )
    fig = px.bar(
        metric_df,
        x='Metric',
        y='Value',
        title='Model Evaluation Metrics',
        text='Value',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


