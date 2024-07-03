from datetime import datetime
import pandas as pd
from batch import prepare_data


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    columns = [
        "PULocationID",
        "DOLocationID",
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
    ]
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    df = pd.DataFrame(data, columns=columns)

    result = prepare_data(df, ["PULocationID", "DOLocationID"])

    expected_result_data = [
        ("-1", "-1", dt(1, 1), dt(1, 10), 9.0),
        ("1", "1", dt(1, 2), dt(1, 10), 8.0),
    ]
    expected_result_df = pd.DataFrame(
        expected_result_data, columns=columns + ["duration"]
    )

    pd.testing.assert_frame_equal(result, expected_result_df)


