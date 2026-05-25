import numpy as np
import pandas as pd
import plotly.express as px


def correlation_heatmap(df: pd.DataFrame):
    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        title='Correlation Heatmap',
    )
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
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
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    if importance is None:
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
