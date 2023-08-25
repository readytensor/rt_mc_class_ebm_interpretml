from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Selects or drops specified columns."""

    def __init__(self, columns: List[str], selector_type: str = "keep"):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        assert selector_type in ["keep", "drop"]
        self.selector_type = selector_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op

        Returns:
            self
        """
        return self

    def transform(self, X: pd.DataFrame):
        """
        Applies the column selection.

        Args:
            X : pandas.DataFrame - The input data.
        Returns:
            pandas.DataFrame: The transformed data.
        """
        if self.selector_type == "keep":
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == "drop":
            dropped_cols = [col for col in X.columns if col in self.columns]
            X = X.drop(dropped_cols, axis=1)
        return X


class DropAllNaNFeatures(BaseEstimator, TransformerMixin):
    """Drops features that contain all NaN values."""

    def __init__(self, columns: List[str]):
        """
        Initializes a new instance of the `ColumnSelector` class.

        Args:
            columns : list of str
                List of column names to select or drop.
            selector_type : str, optional (default='keep')
                Type of selection. Must be either 'keep' or 'drop'.
        """
        self.columns = columns
        self.nan_columns_to_drop = []

    def fit(self, X: pd.DataFrame, y=None):
        cols_to_check = [c for c in X.columns if c in self.columns]
        if len(cols_to_check) == 0:
            return self
        self.nan_columns_to_drop = (
            X[cols_to_check].columns[X[cols_to_check].isna().all()].tolist()
        )
        return self

    def transform(self, X: pd.DataFrame):
        """
        Drops the features that contain all NaN values from the input data.

        Args:
            X : pandas.DataFrame - The input data.

        Returns:
            pandas.DataFrame: The transformed data.
        """
        if len(self.nan_columns_to_drop) > 0:
            X = X.drop(columns=self.nan_columns_to_drop, errors="ignore")
            if X.empty:
                raise ValueError(
                    "All features in the input dataframe are NaN. \
                    Dropping all yields empty dataframe"
                )
        return X


class TypeCaster(BaseEstimator, TransformerMixin):
    """
    A custom transformer that casts the specified variables in the input data
    to a specified data type.
    """

    def __init__(self, vars: List[str], cast_type: str):
        """
        Initializes a new instance of the `TypeCaster` class.

        Args:
            vars : list of str
                List of variable names to be transformed.
            cast_type : data type
                Data type to which the specified variables will be cast.
        """
        super().__init__()
        self.vars = vars
        self.cast_type = cast_type

    def fit(self, X: pd.DataFrame, y=None):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Applies the casting to given features in input dataframe.

        Args:
            data : pandas DataFrame
                Input data to be transformed.
        Returns:
            data : pandas DataFrame
                Transformed data.
        """
        data = data.copy()
        applied_cols = [col for col in self.vars if col in data.columns]
        for var in applied_cols:
            if data[var].notnull().any():  # check if the column has any non-null values
                data[var] = data[var].apply(self.cast_type)
            else:
                # all values are null. so no-op
                pass
        return data


class TransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        transformer: Union[BaseEstimator, TransformerMixin],
        variables: List[str],
        **kwargs,
    ):
        """
        Wrapper class that fits/transforms using given transformer if given variables
        are present in data, else returns the data as-is.

        Args:
            transformer : transformer class
                transformer to apply on categorical features.
            variables : list of str
                List of the features to transform.
            **kwargs : any
                Additional key-value pairs for arguments accepted by the given
                transformer

        """
        self.transformer = transformer
        self.variables = variables
        self.kwargs = kwargs

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the transformer if categorical variables are present.

        Args:
            X: pandas DataFrame - the input data
            y: unused
        Returns:
            self
        """
        self.fitted_vars = list(set(self.variables).intersection(X.columns))
        if len(self.fitted_vars) > 0:
            self.transformer = self.transformer(**self.kwargs)
            self.transformer.fit(X[self.fitted_vars], y)
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Transform the data if categorical variables are present..

        Args:
            X: pandas DataFrame - The data to transform.
            y: unused
        Returns:
            pandas DataFrame - The transformed data with the fitted
                categoricalfeatures.
        """
        if len(self.fitted_vars) == 0:
            return X

        non_fitted_vars = [f for f in X.columns if f not in self.fitted_vars]
        transformed = self.transformer.transform(X[self.fitted_vars])

        # If the transformed data is a numpy array, convert it to a DataFrame
        if isinstance(transformed, np.ndarray):
            transformed = pd.DataFrame(
                transformed, columns=self.fitted_vars, index=X.index
            )

        new_X = pd.concat([X.drop(self.fitted_vars, axis=1), transformed], axis=1)
        new_X.columns = list(non_fitted_vars) + list(transformed.columns)
        return new_X


class ValueClipper(BaseEstimator, TransformerMixin):
    """Clips the values of the specified fields to a specified range."""

    def __init__(
        self,
        fields_to_clip: List[str],
        min_val: Union[float, None],
        max_val: Union[float, None],
    ) -> None:
        """
        Initializes a new instance of the `ValueClipper` class.

        Args:
            fields_to_clip : list of str
                List of field names to clip.
            min_val : float or None, optional (default=None)
                Minimum value of the range.
                If None, the values are not clipped from the lower end.
            max_val : float or None, optional (default=None)
                Maximum value of the range.
                If None, the values are not clipped from the upper end.

        """
        super().__init__()
        self.fields_to_clip = fields_to_clip
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, data: pd.DataFrame):
        """
        No-op.

        Returns:
            self
        """
        return self

    def transform(self, data: pd.DataFrame):
        """
        Clips the values of the specified fields to the specified range.

        Args:
            data: pandas.DataFrame
                The input data.
        Returns:
            pandas.DataFrame
                The transformed data.

        """
        for field in self.fields_to_clip:
            if field not in data.columns:
                continue
            if self.min_val is not None:
                data[field] = data[field].clip(lower=self.min_val)
            if self.max_val is not None:
                data[field] = data[field].clip(upper=self.max_val)
        return data


class MostFrequentImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values using the most frequently observed class for
    categorical features when missing values are rare (under 10% of samples).
    """

    def __init__(self, cat_vars: List[str], threshold: float):
        """
        Initializes a new instance of the `MostFrequentImputer` class.

        Args:
            cat_vars : list of str
                List of the categorical features to impute.
            threshold : float, optional (default=1)
                The minimum proportion of the samples that must contain a missing value
                for the imputation to be performed.

        """
        self.cat_vars = cat_vars
        self.threshold = threshold
        self.fill_vals = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fits the transformer.

        Args:
            X: pandas DataFrame
                The input data
            y: unused
        Returns:
            self
        """
        if self.cat_vars and len(self.cat_vars) > 0:
            self.fitted_cat_vars = [
                var
                for var in self.cat_vars
                if var in X.columns and X[var].isnull().mean() < self.threshold
            ]

            for col in self.fitted_cat_vars:
                self.fill_vals[col] = X[col].value_counts().index[0]
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Transform the data by imputing the most frequent class for the fitted
        categorical features.

        Args:
            X: pandas DataFrame
                The data to transform.
            y: unused
        Returns:
            pandas DataFrame - The transformed data with the most frequent class
                imputed for the fitted categorical features.
        """
        for col in self.fill_vals:
            if col in X.columns:
                X[col] = X[col].fillna(self.fill_vals[col])
        return X


class OneHotEncoderMultipleCols(BaseEstimator, TransformerMixin):
    """Encodes categorical features using one-hot encoding."""

    def __init__(
        self,
        ohe_columns: List[str],
        max_num_categories: int = 10,
        drop_original: bool = True,
    ):
        """
        Initialize a new instance of the `OneHotEncoderMultipleCols` class.

        Args:
            ohe_columns (list[str]): List of the categorical features to one-hot encode.
            max_num_categories (int, optional): Maximum number of categories to include
                for each feature.
            drop_original(bool, optional): Flag to drop or keep the original OHE columns
        """
        super().__init__()
        self.ohe_columns = ohe_columns
        self.max_num_categories = max_num_categories
        self.drop_original = drop_original
        self.is_fitted = False
        self.top_cat_by_ohe_col = {}
        self.fitted_vars = []

    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn the values to be used for one-hot encoding from the input data X.

        Args:
            X (pandas.DataFrame): Data to learn one-hot encoding from.
            y : unused

        Returns:
            OneHotEncoderMultipleCols: self
        """
        self.fitted_vars = list(set(self.ohe_columns).intersection(X.columns))
        for col in self.fitted_vars:
            top_categories = X[col].value_counts().sort_values(ascending=False).index
            # we will drop the redundant last category, and also cap
            # to max_num_categories
            num_categories_to_use = max(
                1, min(self.max_num_categories, len(top_categories) - 1)
            )
            self.top_cat_by_ohe_col[col] = list(top_categories)[:num_categories_to_use]
        return self

    def transform(self, data: pd.DataFrame):
        """
        Encode the input data using the learned values.

        Args:
            data (pandas.DataFrame): Data to one-hot encode.

        Returns:
            transformed_data (pandas.DataFrame): One-hot encoded data.
        """
        if len(self.fitted_vars) == 0:
            return data

        data.reset_index(inplace=True, drop=True)
        df_list = [data]
        cols_list = list(data.columns)

        for col in self.fitted_vars:
            if col not in data.columns:
                raise ValueError(
                    f"Fitted one-hot-encoded column {col} does not exist in dataframe \
                        given for transformation. This will result in a shape \
                            mismatch for train/prediction job."
                )

            for cat in self.top_cat_by_ohe_col[col]:
                col_name = f"{col}_{cat}"
                vals = np.where(data[col] == cat, 1, 0)
                df = pd.DataFrame(vals, columns=[col_name])
                df_list.append(df)
                cols_list.append(col_name)

        transformed_data = pd.concat(df_list, axis=1, ignore_index=True)
        transformed_data.columns = cols_list

        if self.drop_original:
            transformed_data.drop(self.fitted_vars, axis=1, inplace=True)
        return transformed_data


class ColumnOrderTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that keeps the columns in the same order
    to ensure reproducibility.
    """

    def __init__(self):
        """
        Initializes a new instance of the `ColumnOrderTransformer` class.
        """
        super().__init__()
        self.column_order = None

    def fit(self, X: pd.DataFrame, y=None):
        """
        Memorizes the order of DataFrame columns.

        Returns:
            self
        """
        self.column_order = sorted(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame):
        """
        Re-arranges the DataFrame columns to match the original sorted order.

        Args:
            X : pandas DataFrame
                Input data to be transformed.
        Returns:
            X : pandas DataFrame
                Transformed data.
        """
        # This will raise a KeyError if any of the original columns are missing
        X = X[self.column_order]
        return X
