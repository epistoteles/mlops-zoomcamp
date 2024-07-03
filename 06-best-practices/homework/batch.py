#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pickle
import pandas as pd


def get_input_path(year, month):
    default_input_pattern = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
    input_pattern = os.getenv("INPUT_FILE_PATTERN", default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = "s3://epistoteles.nyc-duration/year={year:04d}/month={month:02d}/predictions.parquet"
    output_pattern = os.getenv("OUTPUT_FILE_PATTERN", default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename: str):
    """
    read parquet file
    """
    endpoint_url = os.getenv("S3_ENDPOINT_URL", None)
    options = {"client_kwargs": {"endpoint_url": endpoint_url}} if endpoint_url else {}
    df = pd.read_parquet(f"s3://epistoteles.nyc-duration/{filename}", storage_options=options)
    return df


def write_data(filename: str, df: pd.DataFrame):
    """
    write parquet file
    """
    endpoint_url = os.getenv("S3_ENDPOINT_URL", None)
    options = {"client_kwargs": {"endpoint_url": endpoint_url}} if endpoint_url else {}
    df.to_parquet(
        f"s3://epistoteles.nyc-duration/{filename}",
        engine="pyarrow",
        index=False,
        storage_options=options,
    )


def prepare_data(df: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """
    Prepares the data in the following steps:
    1. Adds a duration column in minutes
    2. Removes rows where duration is less than 1 or greater than 60 minutes
    3. Fills missing values with -1
    """
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df["duration"] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype("int").astype("str")
    return df


def main(year, month, input_file, output_file):
    with open("model.bin", "rb") as f_in:
        dv, lr = pickle.load(f_in)

    categorical = ["PULocationID", "DOLocationID"]

    df = read_data(filename=input_file)
    df = prepare_data(df, categorical=categorical)
    df["ride_id"] = f"{year:04d}/{month:02d}_" + df.index.astype("str")

    dicts = df[categorical].to_dict(orient="records")
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print("predicted mean duration:", y_pred.mean())

    df_result = pd.DataFrame()
    df_result["ride_id"] = df["ride_id"]
    df_result["predicted_duration"] = y_pred

    write_data(filename=output_file, df=df_result)


if __name__ == "__main__":
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    main(year, month, input_file, output_file)
