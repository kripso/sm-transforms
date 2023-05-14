import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import DataType, SaveMode, configure, limit_rate, threaded, transform_df, Input, Output
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import SparkSession
from datetime import datetime
from tweepy import API
import tweepy
from utils.spark_logger import SparkLogger


# must stay 3
NUMBER_OF_PARALLER_REQUEST = 8
DATETIME_NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger = SparkLogger("Retrieve_Twitter_Accounts")

SCHEMA = T.StructType(
    [
        T.StructField("pid", T.StringType()),
        T.StructField("searched_id", T.StringType()),
        T.StructField("searched_name", T.StringType()),
        T.StructField("id", T.LongType()),
        T.StructField("name", T.StringType()),
        T.StructField("screen_name", T.StringType()),
        T.StructField("protected", T.BooleanType()),
        T.StructField("location", T.StringType()),
        T.StructField("description", T.StringType()),
        T.StructField("followers_count", T.IntegerType()),
        T.StructField("friends_count", T.IntegerType()),
        T.StructField("favourites_count", T.IntegerType()),
        T.StructField("statuses_count", T.IntegerType()),
        T.StructField("verified", T.BooleanType()),
        T.StructField("profile_image_url", T.StringType()),
        T.StructField("created_at", T.TimestampType()),
        T.StructField("error", T.StringType()),
    ]
)


@limit_rate(seconds=2)
@threaded(max_workers=5, flatten=True)
def retrieve_accounts(row, tokens):
    auth = tweepy.OAuth2BearerHandler(tokens[row["pid"] % NUMBER_OF_PARALLER_REQUEST])
    twitter_api_v1: API = tweepy.API(auth)
    results = []
    result = {
        "pid": row["pid"],
        "searched_id": row["searched_id"],
        "searched_name": row["searched_name"],
    }
    try:
        accounts = twitter_api_v1.search_users(q=row["searched_name"], count=100)
    except BaseException as err:
        results.append(result | {'error': str(err)})
    else:
        for account in accounts:
            tmp_result = {}
            for key in SCHEMA.fieldNames():
                tmp_result[key] = account._json.get(key)

            tmp_result["created_at"] = datetime.strptime(account._json.get("created_at"), "%a %b %d %H:%M:%S %z %Y")

            results.append(tmp_result | result)

    return results


@configure(master=f"local[{NUMBER_OF_PARALLER_REQUEST}]")
@transform_df(
    Output("/data/twitter/sm-scraps-data/datasets/clean/EU.twitter_accounts", save_mode=SaveMode.APPEND),
    credentials=Input("/data/twitter/sm-scraps-data/datasets/raw/credentials.json", data_type=DataType.JSON),
    # df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/targets.simple"),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/EU"),
)
def compute(ctx: SparkSession, df_in: DataFrame, credentials: DataFrame):
    tokens = [item[0] for item in credentials.select("Bearer_Token").collect()]

    # df = (
    #     df_in
    #     .drop(
    #         "first_seen",
    #         "last_seen",
    #         "identifiers",
    #     ).filter(F.array_contains(F.col("countries"), "gb") | F.array_contains(F.col("countries"), "us"))
    # )

    df = df_in.select(
        F.col("id").alias("searched_id"),
        F.col("name").alias("searched_name"),
    )

    df = (
        df
        .repartition(128)
        .withColumn("pid", F.spark_partition_id())
        .rdd
        .mapPartitions(lambda partition: retrieve_accounts(partition, tokens))
        .toDF(SCHEMA)
    )

    return df
