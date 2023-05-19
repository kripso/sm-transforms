import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from utils.pyspark_utils import configure, transform, Input, Output
from pyspark.sql import functions as F
from pyspark.sql import types as T
from utils.spark_logger import SparkLogger
import json
import re

logger = SparkLogger("metadata_extraction_tweets")
schema = T.StructType([
    T.StructField('start', T.StringType()),
    T.StructField('end', T.StringType()),
    T.StructField('id', T.StringType()),
    T.StructField('username', T.StringType()),
])


@F.udf(returnType=T.ArrayType(T.MapType(T.StringType(), T.StringType())))
def parse_json_string(json_string: str) -> any:
    results = []

    if json_string is not None:
        string_ = re.search(r"mentions=\[([^]]+)]", json_string)
        if string_ is not None:
            return f'{results}'

        string_ = [re.sub(r'(\w+)', r'"\g<1>"', item.replace('=', ':')) for item in s[1].replace(' ', '').split('},{')]
        for item in string_:
            if item[0] != '{':
                item = '{' + item
            if item[-1] != '}':
                item = item + '}'
            results.append(json.loads(item))
    return results


@configure()
@transform(
    out_mentions_count=Output("/data/twitter/sm-scraps-data/datasets/clean/tweets_mentions/mentions_count"),
    out_mentions=Output("/data/twitter/sm-scraps-data/datasets/clean/tweets_mentions/mentions"),
    in_=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets/all", merge_schema=True),
)
def compute(ctx, out_mentions_count: Output, out_mentions: Output, in_: Input):
    in_df = in_.read_df()

    df = (
        in_df
        .select('user_name', 'conversation_id', "entities", 'text')
        .withColumn("entities_parsed", parse_json_string(F.col("entities")))
        .withColumn('entities_exploded', F.explode(F.col('entities_parsed')))
        .withColumn('mentioned_username', F.col('entities_exploded').getItem('username'))
        .withColumn('mentioned_id', F.col('entities_exploded').getItem('id'))
        .drop('entities_parsed', 'entities_exploded', 'entities')
    )

    df_mentions = (
        df
        .groupBy(
            'user_name', 'conversation_id'
        )
        .agg(
            F.first('text').alias('text'),
            F.count('mentioned_username').alias('mentioned_count'),
            F.collect_set('mentioned_username').alias('mentioned_usernames'),
            F.collect_set('mentioned_id').alias('mentioned_ids'),
        )
        .orderBy('mentioned_count', ascending=False)
    )
    out_mentions.write_df(df_mentions)
    df_mentions.show()

    df_mentions_counts = (
        df
        .groupBy(
            'user_name', 'mentioned_username'
        )
        .agg(
            F.collect_set('text').alias('text'),
            F.count('mentioned_username').alias('mentioned_count'),
        )
        .orderBy('mentioned_count', ascending=False)
    )
    out_mentions_count.write_df(df_mentions_counts)
    df_mentions_counts.show()
