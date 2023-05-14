import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import DataType, SaveMode, configure, transform, Input, Output, limit_rate
from concurrent.futures import ThreadPoolExecutor
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

# must stay 3
NUMBER_OF_PARALLER_REQUEST = 7
RETRY_ERRORS = [400, 429, 443, 444, 500, 599]

logger = SparkLogger("Retrieve_Following")

SCHEMA = T.StructType(
    [
        T.StructField("pid", T.LongType()),
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
def _retrieve_accounts(twitter_api_v1: API, followers_name, followers_id, friend_id, pid):
    results = []
    result = {
        "pid": pid,
        "followers_id": followers_id,
        "followers_name": followers_name,
        "error": None,
    }

    try:
        accounts = twitter_api_v1.lookup_users(user_id=friend_id)
    except BaseException as err:
        results.append(result | {"error": str(err)})
        logger.info(pid, "retrieve_account", str(err))
    else:
        for account in accounts:
            tmp_result = {}
            for key in SCHEMA.fieldNames():
                tmp_result[key] = account._json.get(key)
            tmp_result["created_at"] = datetime.strptime(account._json.get("created_at"), "%a %b %d %H:%M:%S %z %Y")

            results.append(tmp_result | result)

    return results


@limit_rate(seconds=60)
def _retrieve_friend_ids(twitter_api_v1: API, name, id, pid):
    result = {
        "pid": pid,
        "name": name,
        "id": id,
        "friend_ids": [],
        "error": None,
    }

    try:
        following_ids = twitter_api_v1.get_friend_ids(user_id=id)
    except BaseException as err:
        result["error"] = str(err)
        logger.info(pid, "retrieve_account", str(err))
    else:
        result["friend_ids"] = following_ids

    return result


def retrieve_friend_ids(auth, partition, pid):
    results = []
    tmp_results = []

    for row in partition:
        twitter_api_v1: API = tweepy.API(auth, wait_on_rate_limit=True, retry_delay=60, retry_count=5, retry_errors=RETRY_ERRORS)
        result = _retrieve_friend_ids(twitter_api_v1, row["name"], row["id"], pid)
        tmp_results.append(result)

        if result.get("error") is not None:
            results.append({"followers_name": row["name"], "followers_id": row["id"], "pid": pid, "error": result.get("error")})

    return results, tmp_results


def retrieve_accounts(auth, tmp_results, pid):
    futures = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for tmp_result in tmp_results:
            for friend_ids in more_itertools.ichunked(tmp_result.get("friend_ids"), 100):
                friend_ids = list(friend_ids)

                twitter_api_v1 = tweepy.API(auth, wait_on_rate_limit=True, retry_delay=5, retry_count=5, retry_errors=RETRY_ERRORS)

                future = executor.submit(lambda: _retrieve_accounts(twitter_api_v1, tmp_result["name"], tmp_result["id"], friend_ids, pid))
                futures.append(future)

                time.sleep(3.2)

        results = more_itertools.flatten([_future.result() for _future in futures])

    return results


def retrieve_accounts_partitioned(partition, tokens):
    partition_result = []

    tmp, partition = itertools.tee(partition)
    pid = tmp.__next__()["pid"]

    auth = tweepy.OAuth2BearerHandler(tokens[pid % NUMBER_OF_PARALLER_REQUEST])

    results, tmp_results = retrieve_friend_ids(auth, partition, pid)
    partition_result.extend(results)

    results = retrieve_accounts(auth, tmp_results, pid)
    partition_result.extend(results)

    return partition_result


@configure(master=f"local[{NUMBER_OF_PARALLER_REQUEST}]")
@transform(
    out_df=Output("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts.following/all", save_mode=SaveMode.APPEND),
    credentials=Input("/data/twitter/sm-scraps-data/datasets/raw/credentials.json", data_type=DataType.JSON),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts/datetime=2022-12-11 21:58:58"),
)
def compute(ctx, out_df: Output, credentials: Input, df_in: Input):
    credentials = credentials.read_df()
    tokens = [item[0] for item in credentials.filter(F.col("App").like("kripso_the_bot_%")).select("Bearer_Token").collect()]

    df: DataFrame = (
        df_in.read_df()
        .filter(F.col("error").isNull())
        .filter(F.col("friends_count") <= 1000)
        .filter(F.col("friends_count") > 20)
        .repartition(128)
        .withColumn("pid", F.spark_partition_id())
    )

    df.show()
    df.summary().show()

    df_out = df.rdd.mapPartitions(lambda partition: retrieve_accounts_partitioned(partition, tokens)).toDF(SCHEMA)

    out_df.write_df(df_out)
