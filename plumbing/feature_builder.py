import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd


def one_hot_encode_categorical(df: pd.DataFrame, cols=list) -> pd.DataFrame:

    df_one_hot = pd.get_dummies(df[cols])
    df = df.drop(cols, axis=1)
    return (df.join(df_one_hot), df_one_hot.columns.tolist())


def binarize_values(df: pd.DataFrame, col_encoding=dict) -> pd.DataFrame:

    for col in col_encoding:
        df.loc[:, col] = df[col].map(col_encoding[col])

    return df


def age_from_days(df: pd.DataFrame, col: str) -> pd.DataFrame:

    df.loc[:, "AGE"] = -(df[col]) // 365
    return df


def select_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:

    return df[cols]
