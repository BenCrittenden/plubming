import json
import pandas as pd


def drop_duplicate_rows(df: pd.DataFrame, cols: list, **kwargs) -> pd.DataFrame:

    return df.drop_duplicates(cols, **kwargs)


def drop_columns_with_null_values(df: pd.DataFrame) -> pd.DataFrame:

    drop_cols = df.columns[df.isnull().any().values]
    print(f"dropping columns with null values: {drop_cols}")
    return df.drop(drop_cols, axis=1)


def drop_columns_with_homogenous_values(df: pd.DataFrame) -> pd.DataFrame:

    homogenous_col = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            homogenous_col += [col]

    return df.drop(homogenous_col, axis=1)
