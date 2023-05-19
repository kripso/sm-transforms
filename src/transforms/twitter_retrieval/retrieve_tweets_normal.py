import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import DataType, SaveMode, configure, transform, Input, Output, limit_rate, threaded
from utils.spark_logger import SparkLogger
from pyspark.sql.dataframe import DataFrame, Row
from pyspark.sql import functions as F
from pyspark.sql import types as T
from tweepy import Client
from typing import List
import tweepy
import json

NUMBER_OF_PARALLER_REQUEST = 1
RETRY_ERRORS = [400, 429, 443, 444, 500, 599]

logger = SparkLogger("Retrieve_Tweets")

# TODO: change entities into proper type
SCHEMA = T.StructType(
    [
        T.StructField("user_name", T.StringType()),
        T.StructField("user_id", T.LongType()),
        T.StructField("screen_name", T.StringType()),
        T.StructField("error", T.StringType()),
        T.StructField("id", T.LongType()),
        T.StructField("text", T.StringType()),
        T.StructField("conversation_id", T.LongType()),
        T.StructField("created_at", T.TimestampType()),
        T.StructField("in_reply_to_user_id", T.LongType()),
        T.StructField("lang", T.StringType()),
        T.StructField(
            "referenced_tweets",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField("type", T.StringType()),
                        T.StructField("id", T.LongType()),
                    ]
                )
            ),
        ),
        T.StructField(
            "context_annotations",
            T.ArrayType(
                T.StructType(
                    [
                        T.StructField(
                            "domain",
                            T.StructType(
                                [
                                    T.StructField("id", T.StringType()),
                                    T.StructField("name", T.StringType()),
                                    T.StructField("description", T.StringType()),
                                ]
                            ),
                        ),
                        T.StructField(
                            "entity",
                            T.StructType(
                                [
                                    T.StructField("id", T.StringType()),
                                    T.StructField("name", T.StringType()),
                                ]
                            ),
                        ),
                    ]
                )
            ),
        ),
        T.StructField("entities", T.StringType()), 
    ]
)

RETRY_ERRORS = [400, 429, 443, 444, 500, 599]
TWEET_KEYS_REQ = [
    "id",
    "author_id",
    "source",
    "text",
    "conversation_id",
    "created_at",
    "entities",
    "in_reply_to_user_id",
    "lang",
    "referenced_tweets",
    "context_annotations",
]
TWEET_KEYS = ["id", "text", "conversation_id", "created_at", "in_reply_to_user_id", "lang"]


@limit_rate(limit=1500, interval_in_secods=900)
def get_tweets(twitter_api_v1: Client, user_id, next_page_token):
    return twitter_api_v1.get_users_tweets(
        id=user_id, tweet_fields=TWEET_KEYS_REQ, exclude=["retweets"], max_results=100, pagination_token=next_page_token
    )


@threaded(max_workers=2, flatten=True)
def retrieve_tweets(row: Row, tokens: str):
    twitter_api_v1: Client = tweepy.Client(bearer_token=tokens[row["pid"] % NUMBER_OF_PARALLER_REQUEST], wait_on_rate_limit=True)

    results: List[Row] = []
    result = {
        "user_name": row["name"],
        "user_id": row["id"],
        "screen_name": row["screen_name"],
    }
    next_page_token = None
    while next_page_token != -1:
        try:
            tweets = get_tweets(twitter_api_v1, row["id"], next_page_token)
        except Exception as err:
            results.append(result | {"error": str(err)})
            logger.info(row["pid"], row["id"], "retrieve_tweets", str(err))
        else:
            if tweets.data is not None:
                for tweet in tweets.data:
                    tmp_result = {}
                    for key in TWEET_KEYS:
                        tmp_result[key] = tweet.get(key)
                    if tweet.get("referenced_tweets") is not None:
                        tmp_result["referenced_tweets"] = [
                            {"id": reference.get("id"), "type": reference.get("type")} for reference in tweet.get("referenced_tweets", [])
                        ]
                    tmp_result["context_annotations"] = tweet.get("context_annotations")
                    tmp_result["entities"] = json.dumps(tweet.get("entities"))
                    results.append(tmp_result | result)
            else:
                results.append(result | {"error": "No Data"})
            next_page_token = tweets.meta.get("next_token", -1)

    return results


@configure(master=f"local[{NUMBER_OF_PARALLER_REQUEST}]")
@transform(
    out_df=Output("/data/twitter/sm-scraps-data/datasets/clean/tweets/all", save_mode=SaveMode.APPEND),
    credentials=Input("/data/twitter/sm-scraps-data/datasets/raw/credentials.json", data_type=DataType.JSON),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts"),
)
def compute(ctx, out_df: Output, credentials: Input, df_in: Input):
    credentials = credentials.read_df()
    # tokens = [item[0] for item in credentials.filter(F.col('App').like('kripso_bot_%')).select("Bearer_Token").collect()]
    tokens = [item[0] for item in credentials.filter(F.col("App").like("Academic")).select("Bearer_Token").collect()]

    df: DataFrame = (
        df_in.read_df()
        .filter(F.col("error").isNull())
        .filter(F.col("friends_count") <= 675)
        .filter(F.col("friends_count") > 670)
        .repartition(64)
        .withColumn("pid", F.spark_partition_id())
    )

    df.show()
    df.summary().show()

    df_out = df.rdd.mapPartitions(lambda partition: retrieve_tweets(partition, tokens)).toDF(SCHEMA)

    out_df.write_df(df_out)
