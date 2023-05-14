import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import configure, transform_df, Input, Output
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from utils.spark_logger import SparkLogger

LOGGER = SparkLogger("Filtered_US_And_UK")


@configure(master="local[*]")
@transform_df(
    Output("/data/twitter/sm-scraps-data/datasets/clean/EU"),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/targets"),
)
def compute(ctx, df_in: DataFrame):
    df_in.drop_duplicates(['id']).summary().show()

    df = (
        df_in
        .drop("first_seen", "last_seen", "identifiers",)
        .filter(
            F.array_contains(F.col("countries"), "be") |
            F.array_contains(F.col("countries"), "bg") |
            F.array_contains(F.col("countries"), "cz") |
            F.array_contains(F.col("countries"), "dk") |
            F.array_contains(F.col("countries"), "de") |
            F.array_contains(F.col("countries"), "ee") |
            F.array_contains(F.col("countries"), "ie") |
            F.array_contains(F.col("countries"), "el") |
            F.array_contains(F.col("countries"), "es") |
            F.array_contains(F.col("countries"), "fr") |
            F.array_contains(F.col("countries"), "hr") |
            F.array_contains(F.col("countries"), "it") |
            F.array_contains(F.col("countries"), "cy") |
            F.array_contains(F.col("countries"), "lv") |
            F.array_contains(F.col("countries"), "lt") |
            F.array_contains(F.col("countries"), "lu") |
            F.array_contains(F.col("countries"), "hu") |
            F.array_contains(F.col("countries"), "mt") |
            F.array_contains(F.col("countries"), "nl") |
            F.array_contains(F.col("countries"), "at") |
            F.array_contains(F.col("countries"), "pl") |
            F.array_contains(F.col("countries"), "pt") |
            F.array_contains(F.col("countries"), "ro") |
            F.array_contains(F.col("countries"), "si") |
            F.array_contains(F.col("countries"), "sk") |
            F.array_contains(F.col("countries"), "fi") |
            F.array_contains(F.col("countries"), "se") |
            F.array_contains(F.col("countries"), "us") |
            F.array_contains(F.col("countries"), "gb") |
            F.array_contains(F.col("countries"), "us")
        )
    )

    for country in [
        "be", "bg", "cz", "dk", "de", "ee", "ie", "gr", "es", "fr",
        "hr", "it", "cy", "lv", "lt", "lu", "hu", "mt", "nl", "at",
        "pl", "pt", "ro", "si", "sk", "fi", "se", 'us', 'gb', 'us']:

        tmp_df = df.filter(F.array_contains(F.col("countries"), country)).dropDuplicates(subset=["countries"])
        print("-" * 200)
        print(country)
        tmp_df.summary().show()

    df.summary().show()

    return df
