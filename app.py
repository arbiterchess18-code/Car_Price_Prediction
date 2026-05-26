import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import (
    clean_missing_values,
    remove_outliers,
    build_preprocessing_pipeline,
    extract_feature_names,
    prepare_brand_feature,
    split_features_target,
    find_manual_input_columns,
)
from utils.visualization import (
    correlation_heatmap,
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_pairplot,
    plot_feature_importance,
    plot_metric_comparison,
    plot_actual_vs_predicted,
)
from utils.modeling import (
    auto_train_best_model,
    get_regression_model,
    get_hyperparameter_grid,
    train_model_pipeline,
)

st.set_page_config(
    page_title='AI Car Price Prediction',
    layout='wide',
    page_icon='🚗',
)


def init_session_state():
    defaults = {
        'df_raw': None,
        'df_clean': None,
        'target_column': None,
        'preprocessor': None,
        'model_pipeline': None,
        'metrics': None,
        'feature_names': None,
        'trained_model': None,
        'model_name': None,
        'preprocessed_X': None,
        'preprocessed_y': None,
        'manual_columns': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_dataset_summary(df: pd.DataFrame):
    st.subheader('Dataset Preview')
    st.dataframe(df.head(7), width='stretch')
    st.write('Shape:', df.shape)
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if missing.empty:
        st.success('No missing values detected.')
    else:
        st.warning('Missing values found in dataset.')
        st.dataframe(missing.to_frame('Missing Count'))
    st.markdown('**Columns:**')
    st.write(list(df.columns))


def show_upload_section():
    st.header('1. Dataset Upload')
    uploaded_file = st.file_uploader('Upload your car dataset as CSV', type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df_raw = df.copy()
            st.session_state.df_clean = df.copy()
            st.session_state.target_column = None
            st.session_state.preprocessor = None
            st.session_state.model_pipeline = None
            st.session_state.metrics = None
            st.session_state.trained_model = None
            st.session_state.feature_names = None
            st.session_state.preprocessed_X = None
            st.session_state.preprocessed_y = None
            st.session_state.manual_columns = None
            st.success('Dataset uploaded successfully!')
            display_dataset_summary(df)
            st.markdown('---')
            target_columns = list(df.columns)
            default_target = 'selling_price' if 'selling_price' in target_columns else next(
                (col for col in target_columns if 'price' in col.lower()), target_columns[0]
            )
            target_option = st.selectbox(
                'Select the target column to predict',
                options=target_columns,
                index=target_columns.index(default_target),
            )
            if target_option:
                st.session_state.target_column = target_option
                if target_option.lower() == 'name':
                    st.warning(
                        'The dataset name field should not be used as target. Choose a price column like selling_price.'
                    )
                else:
                    st.info(f'Selected target column: **{target_option}**')
        except Exception as ex:
            st.error('Unable to read CSV file. Please upload a valid CSV.')
            st.error(str(ex))
    else:
        st.info('Upload a CSV file to begin the ML workflow.')


def show_preprocessing_section():
    st.header('2. Preprocessing')
    if st.session_state.df_raw is None:
        st.warning('Upload a dataset first in the Dataset Upload section.')
        return

    df = st.session_state.df_raw.copy()
    st.subheader('Data Cleaning Options')
    missing_strategy = st.selectbox(
        'Missing value strategy',
        options=['None', 'Drop rows', 'Mean/Mode Imputation', 'Median/Mode Imputation'],
        index=2,
    )
    encoding_choice = st.selectbox(
        'Categorical encoding',
        options=['None', 'Label Encoding', 'One-Hot Encoding'],
        index=2,
    )
    scaling_choice = st.selectbox(
        'Feature scaling',
        options=['None', 'Standard Scaling', 'MinMax Scaling'],
        index=1,
    )
    remove_outliers_flag = st.checkbox('Remove outliers from numeric features', value=False)

    if st.button('Apply preprocessing'):
        with st.spinner('Applying preprocessing pipeline...'):
            if st.session_state.target_column is None:
                st.error('Please select a target column in the Upload section.')
                return

            if st.session_state.target_column.lower() == 'name':
                if 'selling_price' in df.columns:
                    new_target = 'selling_price'
                else:
                    new_target = next((col for col in df.columns if 'price' in col.lower()), None)
                if new_target:
                    st.warning(f"Target 'name' is not valid. Switching target to '{new_target}'.")
                    st.session_state.target_column = new_target
                else:
                    st.error('No valid price target available. Please upload a dataset with a price column.')
                    return

            if 'name' in df.columns and 'brand' not in df.columns:
                df = prepare_brand_feature(df)
                st.info('Extracted car brand from name and added a new brand feature.')

            if missing_strategy != 'None':
                df = clean_missing_values(df, missing_strategy)
                st.info(f'Applied missing value strategy: {missing_strategy}')

            if remove_outliers_flag:
                numeric_cols_for_outliers = df.select_dtypes(include=np.number).columns.tolist()
                if st.session_state.target_column in numeric_cols_for_outliers:
                    numeric_cols_for_outliers.remove(st.session_state.target_column)
                df = remove_outliers(df, numeric_cols_for_outliers)
                st.info('Removed outliers from numeric features.')

            if st.session_state.target_column not in df.columns:
                st.error(f"Target column '{st.session_state.target_column}' not found in dataset after preprocessing.")
                return

            try:
                X, y = split_features_target(df, st.session_state.target_column)
            except ValueError as ex:
                st.error(str(ex))
                return
            preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(
                X,
                encoding=encoding_choice,
                scaling=scaling_choice,
            )
            X_processed = preprocessor.fit_transform(X)
            feature_names = extract_feature_names(preprocessor, numeric_cols, categorical_cols)

            st.session_state.df_clean = df
            st.session_state.preprocessor = preprocessor
            st.session_state.feature_names = feature_names
            st.session_state.preprocessed_X = X_processed
            st.session_state.preprocessed_y = y.values
            st.session_state.manual_columns = find_manual_input_columns(df)

            st.success('Preprocessing completed successfully.')
            st.write('Processed feature count:', X_processed.shape[1])
            st.write('Data shape after preprocessing:', df.shape)

    if st.session_state.df_clean is not None:
        st.markdown('### Current cleaned dataset preview')
        st.dataframe(st.session_state.df_clean.head(5), width='stretch')


def show_eda_section():
    st.header('3. EDA & Visualization')
    if st.session_state.df_clean is None:
        st.warning('Please upload and preprocess a dataset first.')
        return

    df = st.session_state.df_clean.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.subheader('Select visualizations to display')
    col1, col2, col3 = st.columns(3)
    with col1:
        show_histogram = st.checkbox('Histogram', value=True)
        show_boxplot = st.checkbox('Boxplot', value=True)
        show_correlation = st.checkbox('Correlation Heatmap', value=True)
    with col2:
        show_scatter = st.checkbox('Scatter Plot', value=True)
        show_pairplot = st.checkbox('Pairplot', value=True)
        show_feature_imp = st.checkbox('Feature Importance', value=True)
    with col3:
        st.write('')

    st.subheader('Dataset Visualizations')
    
    if show_histogram:
        col1, col2 = st.columns(2)
        with col1:
            chart_columns = st.multiselect('Choose columns for distribution charts', numeric_cols, default=numeric_cols[:3], key='hist_columns')
            if chart_columns:
                st.plotly_chart(plot_histogram(df, chart_columns), use_container_width=True)
    
    if show_boxplot:
        col1, col2 = st.columns(2)
        with col1:
            box_columns = st.multiselect('Choose columns for boxplots', numeric_cols, default=numeric_cols[:3], key='box_columns')
            if box_columns:
                st.plotly_chart(plot_boxplot(df, box_columns), use_container_width=True)

    st.markdown('---')
    st.subheader('Correlation and Scatter Analyses')
    
    if show_correlation:
        if numeric_cols:
            st.plotly_chart(correlation_heatmap(df[numeric_cols]), use_container_width=True)

    if show_scatter:
        scatter_x = st.selectbox('Scatter X axis', numeric_cols, index=0 if numeric_cols else None, key='scatter_x')
        scatter_y = st.selectbox('Scatter Y axis', numeric_cols, index=1 if len(numeric_cols) > 1 else None, key='scatter_y')
        if scatter_x and scatter_y and scatter_x != scatter_y:
            st.plotly_chart(plot_scatter(df, scatter_x, scatter_y), use_container_width=True)

    if show_pairplot:
        if len(numeric_cols) >= 3:
            selected_pair = st.multiselect('Select features for pairplot', numeric_cols[:5], default=numeric_cols[:3], key='pairplot_columns')
            if selected_pair:
                st.plotly_chart(plot_pairplot(df, selected_pair), use_container_width=True)

    if show_feature_imp:
        if st.session_state.trained_model is not None and st.session_state.feature_names is not None:
            st.markdown('---')
            st.subheader('Feature importance')
            importance_chart = plot_feature_importance(
                st.session_state.trained_model,
                st.session_state.feature_names,
            )
            if importance_chart is not None:
                st.plotly_chart(importance_chart, use_container_width=True)
            else:
                st.info('Feature importance is unavailable for the selected model.')


def show_modeling_section():
    st.header('4. Model Selection & Training')
    if st.session_state.df_clean is None:
        st.warning('Upload and preprocess a dataset first.')
        return
    if st.session_state.target_column is None:
        st.error('Please select a target column in the Upload section.')
        return

    df = st.session_state.df_clean.copy()
    X, y = split_features_target(df, st.session_state.target_column)

    model_choice = st.selectbox(
        'Choose regression model',
        ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'KNN Regressor', 'XGBoost Regressor'],
    )
    split_ratio = st.slider('Train test split ratio', min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.number_input('Random state', min_value=0, max_value=9999, value=42, step=1)
    use_cross_validation = st.checkbox('Enable cross-validation', value=False)
    use_hyperparam_search = st.checkbox('Use hyperparameter tuning', value=False)
    search_type = None
    if use_hyperparam_search:
        search_type = st.selectbox('Search method', ['GridSearchCV', 'RandomizedSearchCV'])

    col1, col2 = st.columns(2)
    with col1:
        if st.button('Train model'):
            with st.spinner('Training the model. This may take a moment...'):
                preprocessor = st.session_state.preprocessor
                if preprocessor is None:
                    st.warning('No preprocessing pipeline found. Rebuilding with default options.')
                    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(X)
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = extract_feature_names(preprocessor, numeric_cols, categorical_cols)
                model = get_regression_model(model_choice)
                param_grid = get_hyperparameter_grid(model_choice)
                model_pipeline, metrics = train_model_pipeline(
                    X,
                    y,
                    preprocessor,
                    model,
                    param_grid if use_hyperparam_search else None,
                    search_type,
                    split_ratio,
                    random_state,
                    use_cross_validation,
                )
                st.session_state.model_pipeline = model_pipeline
                st.session_state.trained_model = model_pipeline.named_steps['model']
                st.session_state.metrics = metrics
                st.session_state.model_name = model_choice
    with col2:
        if st.button('Auto-Train Best Model'):
            with st.spinner('Searching for the best model automatically...'):
                preprocessor = st.session_state.preprocessor
                if preprocessor is None:
                    st.warning('No preprocessing pipeline found. Rebuilding with default options.')
                    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(X)
                    st.session_state.preprocessor = preprocessor
                    st.session_state.feature_names = extract_feature_names(preprocessor, numeric_cols, categorical_cols)

                best_pipeline, best_metrics, best_model_name = auto_train_best_model(
                    X,
                    y,
                    preprocessor,
                    split_ratio,
                    random_state,
                    use_cross_validation,
                    use_hyperparam_search,
                    search_type,
                )
                if best_pipeline is None or best_metrics is None:
                    st.error('Unable to auto-train any model successfully.')
                else:
                    st.session_state.model_pipeline = best_pipeline
                    st.session_state.trained_model = best_pipeline.named_steps['model']
                    st.session_state.metrics = best_metrics
                    st.session_state.model_name = best_model_name
                    st.success(f'Auto-trained best model: {best_model_name}')

    if st.session_state.metrics:
        st.subheader('Model Evaluation Metrics')
        metrics = st.session_state.metrics
        st.metric('MAE', f'{metrics.get("mae", 0):.2f}')
        st.metric('MSE', f'{metrics.get("mse", 0):.2f}')
        st.metric('RMSE', f'{metrics.get("rmse", 0):.2f}')
        st.metric('R² Score', f'{metrics.get("r2", 0):.2f}')

        st.markdown('### Actual vs Predicted')
        st.plotly_chart(plot_actual_vs_predicted(metrics.get('y_test'), metrics.get('y_pred')), use_container_width=True)

        st.markdown('### Model Comparison')
        st.plotly_chart(plot_metric_comparison(metrics), use_container_width=True)

        if metrics.get('best_params'):
            st.markdown('### Best Hyperparameters')
            st.json(metrics.get('best_params'))


def show_prediction_section():
    st.header('5. Manual Prediction')
    if st.session_state.model_pipeline is None:
        st.warning('Train a model first in the Model Selection section.')
        return
    if st.session_state.manual_columns is None:
        st.warning('Manual input fields could not be inferred from your dataset.')
        return

    manual_cols = st.session_state.manual_columns
    st.subheader('Enter car details for prediction')
    input_data = {}
    for key, col_info in manual_cols.items():
        label = col_info['label']
        if col_info['type'] == 'numeric':
            value = st.number_input(label, value=col_info.get('default', 0.0), step=1.0)
        else:
            choices = col_info.get('options')
            if choices:
                value = st.selectbox(label, options=choices)
            else:
                value = st.text_input(label, value=col_info.get('default', ''))
        input_data[col_info['name']] = value

    if st.button('Predict Price'):
        with st.spinner('Predicting car price...'):
            all_features = st.session_state.df_clean.drop(columns=[st.session_state.target_column]).columns.tolist()
            manual_row = {feature: None for feature in all_features}
            manual_row.update(input_data)
            for feature in all_features:
                if manual_row[feature] is None:
                    if pd.api.types.is_numeric_dtype(st.session_state.df_clean[feature]):
                        manual_row[feature] = float(st.session_state.df_clean[feature].median())
                    else:
                        manual_row[feature] = str(st.session_state.df_clean[feature].mode().iloc[0])
            row = pd.DataFrame([manual_row])
            try:
                prediction = st.session_state.model_pipeline.predict(row)
                st.success(f'Estimated car price: ₹{prediction[0]:,.2f}')
            except Exception as ex:
                st.error('Unable to generate prediction from the entered values.')
                st.error(str(ex))


def main():
    init_session_state()
    st.title('🚗 AI-Powered Car Price Prediction')
    st.markdown(
        'An interactive machine learning dashboard for car price prediction with dataset upload, preprocessing, model training, and instant predictions.'
    )

    sidebar = st.sidebar
    sidebar.title('Navigation')
    page = sidebar.radio(
        'Go to',
        ['Dataset Upload', 'Preprocessing', 'EDA & Visualization', 'Modeling', 'Manual Prediction'],
    )
    sidebar.markdown('---')
    sidebar.write('Use this application to upload your car dataset, preprocess it, explore features, train a model, and make instant predictions.')

    if page == 'Dataset Upload':
        show_upload_section()
    elif page == 'Preprocessing':
        show_preprocessing_section()
    elif page == 'EDA & Visualization':
        show_eda_section()
    elif page == 'Modeling':
        show_modeling_section()
    elif page == 'Manual Prediction':
        show_prediction_section()


if __name__ == '__main__':
    main()
