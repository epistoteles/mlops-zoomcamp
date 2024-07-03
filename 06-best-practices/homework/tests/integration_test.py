import os
import pandas as pd
from datetime import datetime
from batch import write_data, read_data, main



def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_s3_storage():
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"
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

    write_data("data.parquet", df)

    data = read_data("data.parquet")

    pd.testing.assert_frame_equal(df, data)


def test_batch_process():
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"
    input_file = "data.parquet"
    output_file = "predictions.parquet"
    main(2023, 1, input_file, output_file)

    data = read_data("predictions.parquet")

    print(data['predicted_duration'].sum())