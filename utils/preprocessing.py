import inspect
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def clean_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'Drop rows':
        df_clean = df_clean.dropna()
        return df_clean

    if strategy not in ['Mean/Mode Imputation', 'Median/Mode Imputation']:
        return df_clean

    numeric_strategy = 'mean' if strategy == 'Mean/Mode Imputation' else 'median'
    if num_cols:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        df_clean[num_cols] = num_imputer.fit_transform(df_clean[num_cols])

    if cat_cols:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df_clean[cat_cols] = cat_imputer.fit_transform(df_clean[cat_cols])

    return df_clean


def extract_brand_name(model_name: str) -> str:
    if not isinstance(model_name, str) or not model_name.strip():
        return 'Unknown'

    known_brands = [
        'Mercedes-Benz', 'Land Rover', 'Rolls-Royce', 'BMW', 'Audi', 'Jaguar',
        'Volkswagen', 'Chevrolet', 'Hyundai', 'Honda', 'Maruti', 'Toyota',
        'Tata', 'Ford', 'Renault', 'Nissan', 'Kia', 'MG', 'Skoda', 'Volvo',
        'Datsun', 'Jeep', 'Fiat', 'Mitsubishi', 'Mahindra', 'Bentley', 'Porsche',
    ]
    lower_name = model_name.strip().lower()
    for brand in known_brands:
        if lower_name.startswith(brand.lower()):
            return brand

    first_token = model_name.strip().split()[0]
    return first_token


def prepare_brand_feature(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    if 'name' in df_clean.columns and 'brand' not in df_clean.columns:
        df_clean['brand'] = df_clean['name'].apply(extract_brand_name)
    return df_clean


def remove_outliers(df: pd.DataFrame, numeric_columns: list) -> pd.DataFrame:
    df_clean = df.copy()
    for column in numeric_columns:
        if column not in df_clean.columns:
            continue
        q1 = df_clean[column].quantile(0.25)
        q3 = df_clean[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
    return df_clean


def split_features_target(df: pd.DataFrame, target_column: str):
    if target_column not in df.columns:
        raise ValueError(f'Target column {target_column} not found in dataframe.')
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()
    return X, y


def build_preprocessing_pipeline(
    X: pd.DataFrame,
    encoding: str = 'One-Hot Encoding',
    scaling: str = 'Standard Scaling',
):
    X_clean = X.copy()
    numeric_columns = X_clean.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = X_clean.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_steps = []
    if numeric_columns:
        numeric_steps.append(('imputer', SimpleImputer(strategy='median')))
        if scaling == 'Standard Scaling':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scaling == 'MinMax Scaling':
            numeric_steps.append(('scaler', MinMaxScaler()))

    categorical_steps = []
    if categorical_columns:
        categorical_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
        if encoding == 'One-Hot Encoding':
            encoder_kwargs = {'handle_unknown': 'ignore'}
            if 'sparse_output' in inspect.signature(OneHotEncoder.__init__).parameters:
                encoder_kwargs['sparse_output'] = False
            else:
                encoder_kwargs['sparse'] = False
            categorical_steps.append(
                (
                    'encoder',
                    OneHotEncoder(**encoder_kwargs),
                )
            )
        elif encoding == 'Label Encoding':
            categorical_steps.append(
                (
                    'encoder',
                    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
                )
            )

    transformers = []
    if numeric_columns:
        transformers.append(('num', Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(('cat', Pipeline(categorical_steps), categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', verbose_feature_names_out=False)
    return preprocessor, numeric_columns, categorical_columns


def extract_feature_names(preprocessor: ColumnTransformer, numeric_columns: list, categorical_columns: list) -> list:
    feature_names = []
    if numeric_columns:
        feature_names.extend(numeric_columns)
    if categorical_columns:
        cat_pipeline = preprocessor.named_transformers_.get('cat')
        if cat_pipeline is not None:
            encoder = cat_pipeline.named_steps.get('encoder')
            if hasattr(encoder, 'get_feature_names_out'):
                encoded_names = encoder.get_feature_names_out(categorical_columns)
                feature_names.extend(encoded_names.tolist())
            else:
                feature_names.extend(categorical_columns)
    return feature_names


def _match_column(columns: list, keywords: list, numeric=False):
    lower_cols = {col.lower(): col for col in columns}
    for keyword in keywords:
        for lower_name, original in lower_cols.items():
            if keyword in lower_name:
                return original
    return None


def find_manual_input_columns(df: pd.DataFrame) -> dict:
    columns = list(df.columns)
    field_requirements = {
        'brand': {'keywords': ['brand', 'make', 'manufacturer', 'model', 'name'], 'type': 'categorical'},
        'year': {'keywords': ['year', 'manufacture', 'model_year'], 'type': 'numeric'},
        'fuel_type': {'keywords': ['fuel', 'fuel_type', 'fueltype'], 'type': 'categorical'},
        'transmission': {'keywords': ['transmission', 'trans'], 'type': 'categorical'},
        'seller_type': {'keywords': ['seller', 'seller_type', 'seller type'], 'type': 'categorical'},
        'owner': {'keywords': ['owner'], 'type': 'categorical'},
        'mileage': {'keywords': ['mileage', 'miles', 'mpg'], 'type': 'numeric'},
        'engine_size': {'keywords': ['engine', 'cc', 'displacement'], 'type': 'numeric'},
        'kilometers_driven': {'keywords': ['kilometer', 'km', 'kilometers', 'distance'], 'type': 'numeric'},
    }

    manual_fields = {}
    for field_name, props in field_requirements.items():
        matched = _match_column(columns, props['keywords'], numeric=props['type'] == 'numeric')
        if matched:
            if matched == 'name':
                field_label = 'Name'
            elif field_name == 'brand':
                field_label = 'Brand'
            else:
                field_label = field_name.replace('_', ' ').title()
            field_data = {'name': matched, 'label': f'{field_label} ({matched})', 'type': props['type']}
            if props['type'] == 'categorical':
                unique_values = df[matched].dropna().astype(str).unique().tolist()
                if len(unique_values) > 0 and len(unique_values) <= 500:
                    field_data['options'] = sorted(unique_values)
            if props['type'] == 'numeric':
                default_value = float(df[matched].median()) if pd.api.types.is_numeric_dtype(df[matched]) else 0.0
                field_data['default'] = default_value
            manual_fields[field_name] = field_data

    return manual_fields
