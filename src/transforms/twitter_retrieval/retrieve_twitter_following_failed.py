import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from typing import List
from utils.pyspark_utils import DataType, SaveMode, configure, transform, Input, Output, limit_rate
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from datetime import datetime
from tweepy import API
import more_itertools
import itertools
import tweepy
import time
from utils.spark_logger import SparkLogger
from pyspark.sql import Row

# must stay 3
NUMBER_OF_PARALLER_REQUEST = 7
RETRY_ERRORS = [400, 429, 443, 444, 500, 599]

logger = SparkLogger("Retrieve_Following")

SCHEMA = T.StructType(
    [
        T.StructField("followers_id", T.StringType()),
        T.StructField("followers_name", T.StringType()),
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


@limit_rate(seconds=3)
def _retrieve_accounts(row: Row, twitter_api_v1: API):
    results = []
    result = {
        "followers_id": row["followers_id"],
        "followers_name": row["followers_name"],
        "error": None,
    }

    try:
        accounts = twitter_api_v1.lookup_users(user_id=row["id"])
    except BaseException as err:
        results.append(result | {"error": str(err)})
        logger.info(row["pid"], "retrieve_account", str(err))
    else:
        for account in accounts:
            tmp_result = {}
            for key in SCHEMA.fieldNames():
                tmp_result[key] = account._json.get(key)
            tmp_result["created_at"] = datetime.strptime(account._json.get("created_at"), "%a %b %d %H:%M:%S %z %Y")

            results.append(tmp_result | result)

    return results


def retrieve_accounts(partition, tokens: List[str]):
    futures = []
    tmp, partition = itertools.tee(partition)
    pid = tmp.__next__()["pid"]

    for rows in more_itertools.ichunked(partition, 100):
        auth = tweepy.OAuth2BearerHandler(tokens[pid % NUMBER_OF_PARALLER_REQUEST])
        twitter_api_v1 = tweepy.API(auth, wait_on_rate_limit=True, retry_delay=5, retry_count=5, retry_errors=RETRY_ERRORS)

        futures.append(_retrieve_accounts(twitter_api_v1))

        time.sleep(3.2)

    results = more_itertools.flatten([_future.result() for _future in futures])

    return results


@configure(master=f"local[{NUMBER_OF_PARALLER_REQUEST}]")
@transform(
    out_df=Output("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts.following/all", save_mode=SaveMode.APPEND),
    credentials=Input("/data/twitter/sm-scraps-data/datasets/raw/credentials.json", data_type=DataType.JSON),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts"),
    df_in_failed=Input("/data/twitter/sm-scraps-data/datasets/archive/twitter_accounts.following_v2/all"),
)
def compute(ctx, out_df: Output, credentials: Input, df_in: Input, df_in_failed: Input):
    credentials = credentials.read_df()
    tokens = [item[0] for item in credentials.filter(F.col("App").like("kripso_the_bot_%")).select("Bearer_Token").collect()]

    df_failed: DataFrame = df_in_failed.read_df()

    df: DataFrame = (
        df_in.read_df()
        .filter(F.col("error").isNull())
        .filter(F.col("friends_count") <= 1000)
        .filter(F.col("friends_count") > 20)
        .select(
            F.col("id").alias("followers_id"),
        )
    )

    df = df.join(df_failed, on=["followers_id"]).repartition(128).withColumn("pid", F.spark_partition_id())

    df.show()
    print(df.count())
    # df.summary().show()

    df_out = df.rdd.mapPartitions(lambda partition: retrieve_accounts(partition, tokens)).toDF(SCHEMA)

    out_df.write_df(df_out)
