from raw_transformation import *
from feature_builder import *
from constants import ROOT_DIR
import os


# Pipeline


def load_csv_data(filename):

    # Load the data
    return pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", filename))


def bronze_to_silver(df_applications, df_credit, silver_filename):

    # Bronze to Silver conversion
    df_applications = drop_duplicate_rows(
        df_applications,
        cols=["ID"],
        **dict(keep="first"),
    )
    df_applications = drop_columns_with_null_values(df_applications)
    df_applications = drop_columns_with_homogenous_values(df_applications)

    df_credit = df_credit.drop("MONTHS_BALANCE", axis=1)
    df_credit["APPROVED"] = df_credit["STATUS"].map(
        {
            "C": True,
            "X": True,
            "0": False,  # choosing false here to balance the classes
            "1": False,
            "2": False,
            "3": False,
            "4": False,
            "5": False,
        }
    )

    df_silver_data = pd.merge(df_applications, df_credit, how="inner", on="ID")

    df_silver_data.to_json(
        os.path.join(ROOT_DIR, "data", "preprocessed", silver_filename),
        orient="records",
        indent=2,
    )

    return df_silver_data


def silver_to_gold(silver_filename, gold_filename):

    # Silver to Gold conversion
    df = pd.read_json(
        os.path.join(ROOT_DIR, "data", "preprocessed", silver_filename),
        orient="records",
    )
    df, one_hot_col_names = one_hot_encode_categorical(
        df, cols=["NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE"]
    )
    df = binarize_values(
        df,
        {
            "FLAG_OWN_REALTY": dict(N=False, Y=True),
            "FLAG_OWN_CAR": dict(N=False, Y=True),
            "CODE_GENDER": dict(M=0, F=1),
        },
    )
    df = age_from_days(df, "DAYS_BIRTH")

    specified_columns = [
        "ID",
        "FLAG_OWN_REALTY",
        "FLAG_OWN_CAR",
        "CODE_GENDER",
        "AGE",
        "AMT_INCOME_TOTAL",
        "CNT_FAM_MEMBERS",
        "FLAG_WORK_PHONE",
        "FLAG_PHONE",
        "FLAG_EMAIL",
        "APPROVED",
    ] + one_hot_col_names
    df = select_columns(df, specified_columns)

    df.reset_index(drop=True).to_feather(
        os.path.join(ROOT_DIR, "data", "transformed", gold_filename)
    )

    return df


def run_data_pipeline(
    raw_applications_filename,
    raw_credit_filename,
    silver_filename,
    gold_filename,
    preprocessing_steps,
    feature_building_steps,
):

    df_applications = load_csv_data(raw_applications_filename)
    df_credit = load_csv_data(raw_credit_filename)
    bronze_to_silver(df_applications, df_credit, silver_filename)
    silver_to_gold(silver_filename, gold_filename)
