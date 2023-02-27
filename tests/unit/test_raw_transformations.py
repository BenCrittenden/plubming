import pandas as pd
import pytest

from plumbing.raw_transformation import *


@pytest.mark.parametrize(
    "input_df, test_df",
    [
        (
            pd.DataFrame(
                {
                    "feat1": [
                        1,
                        1,
                        1,
                    ],
                    "feat2": ["h", "h", "h"],
                    "feat3": [False, False, False],
                    "feat4": [1, 2, 3],
                }
            ),
            pd.DataFrame(
                {
                    "feat4": [1, 2, 3],
                }
            ),
        )
    ],
)
def test_drop_columns_with_homogenous_values(input_df, test_df):
    output_df = drop_columns_with_homogenous_values(input_df)
    pd.testing.assert_frame_equal(test_df, output_df)
