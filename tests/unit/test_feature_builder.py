import pandas as pd
import pytest

from plumbing.feature_builder import *


@pytest.mark.parametrize(
    "input_df, col, test_df",
    [
        (
            pd.DataFrame({"days": [-1, -366, -3655]}),
            "days",
            pd.DataFrame({"days": [-1, -366, -3655], "AGE": [0, 1, 10]}),
        )
    ],
)
def test_age_from_days(input_df, col, test_df):
    output_df = age_from_days(input_df, col)
    pd.testing.assert_frame_equal(test_df, output_df)
