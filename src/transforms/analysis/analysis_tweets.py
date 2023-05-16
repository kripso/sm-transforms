import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from utils.pyspark_utils import configure, transform, Input, Output
from utils.spark_logger import SparkLogger
from utils.file_handling import load_dictionary
import matplotlib.pyplot as plt
import numpy as np
import json
import re


logger = SparkLogger("Following_Analysis")

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
        s = re.search(r'mentions=\[([^]]+)]', json_string)
        if s is None:
            return f'{results}'

        s = [re.sub('(\w+)', '"\g<1>"', item.replace('=', ':')) for item in s[1].replace(' ', '').split('},{')]
        for item in s:
            if item[0] != '{':
                item = '{' + item
            if item[-1] != '}':
                item = item + '}'
            results.append(json.loads(item))
    return results


@configure()
@transform(
    out_=Output("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts_analysis", build_datetime=False),
    in_df=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets/all", merge_schema=True),
)
def compute(ctx, out_: Output, in_df: Input):

    in_df: DataFrame = in_df.read_df()
    # urls_df = (
    #     in_df
    #     # .filter(F.col('build_datetime')>=datetime.strptime('2022-12-18 21:00:00', "%Y-%m-%d %H:%M:%S"))
    #     .filter(F.col('entities').contains('.jpg'))
    #     .withColumn("url", F.concat(F.lit("https://"), F.regexp_extract('entities', "\\b(?:display_url=(.[^\s]+))(?:[^ ]+)", 1)))
    #     .withColumn('tweet_url', F.concat(F.lit("https://twitter.com/anyuser/status/"), F.col('conversation_id')))
    #     # .drop_duplicates(subset=['conversation_id'])
    #     .orderBy(F.col("created_at"), ascending=False)
    #     .select('url', 'tweet_url')
    # )
    # urls_df.show(truncate=False)
    # print(urls_df.count())
    relations = load_dictionary('family_v2', ['relatives', 'prefixes'])
    test = [*relations['relatives']]
    for prefix in relations['prefixes']:
        for relative in relations['relatives']:
            test.append(prefix + relative)
    # in_df.drop(
    #     'conversation_id',
    #     'conversation_id',
    #     'user_id',
    #     'build_datetime',
    #     'context_annotations',
    #     'entities',
    #     'in_reply_to_user_id'
    # ).drop_duplicates(subset=['user_name']).show()
    # in_df.count()

    df_with_relations = (
        in_df
        .select('user_name', 'conversation_id', "entities", 'text')
        # whole string match
        .withColumn("text_contains_word", F.col("text").rlike("(?:\\b\\s*)(" + "|".join(test) + ")(?:\\b\\s*|$)"))
        .filter(F.col("text_contains_word"))
        .drop('text_contains_word')
        .withColumn("text_word", F.regexp_extract("text", "(?:\\b\\s*)(" + "|".join(test) + ")(?:\\b\\s*|$)", 1))

        # part of string match
        # .withColumn("text_contains_word", F.col("text").rlike("(^|\w*)(" + "|".join(test) + ")(\w*|$)"))
        # .withColumn("text_word", F.regexp_extract("text", "(^|\w*)(" + "|".join(test) + ")(\w*|$)", 0))
    )
    out_.write_to_html(df_with_relations.drop('text_contains_word', 'entities'), 'relations')

    df = (
        df_with_relations
        .select('user_name', 'conversation_id', "entities", 'text', 'text_word')
        .withColumn("entities_parsed", parse_json_string(F.col("entities")))
        .withColumn('entities_exploded', F.explode(F.col('entities_parsed')))
        .withColumn('mentioned_username', F.col('entities_exploded').getItem('username'))
        .withColumn('mentioned_id', F.col('entities_exploded').getItem('id'))
        .drop('entities_parsed', 'entities_exploded', 'entities')
    )

    df_mentions = (
        df
        .groupBy(
            'conversation_id', 'user_name'
        )
        .agg(
            F.collect_set('text').alias('text'),
            F.count('mentioned_username').alias('mentioned_count'),
            F.collect_set('text_word').alias('text_words'),
            F.collect_set('mentioned_username').alias('mentioned_usernames'),
            F.collect_set('mentioned_id').alias('mentioned_ids'),
        )
        .orderBy('mentioned_count', ascending=False)
        .limit(100)
    )
    out_.write_to_html(df_mentions, 'in_mentions')

    df_mentions_counts = (
        df
        .groupBy(
            'user_name', 'mentioned_username'
        )
        .agg(
            F.collect_set('text').alias('text'),
            F.collect_set('text_word').alias('text_words'),
            F.count('mentioned_username').alias('mentioned_count'),
        )
        .orderBy('mentioned_count', ascending=False)
        .limit(100)
    )
    out_.write_to_html(df_mentions_counts, 'in_mentions_count')

    df = (
        df
        .groupBy('text_word')
        .agg(F.count('conversation_id').alias('count'))
    ).orderBy('count', ascending=False)

    pie_data = [item for item in df.select("text_word", "count").head(10)]
    y = np.array([item[1] for item in pie_data])
    labes = [item[0] for item in pie_data]

    plt.pie(y, labels=labes)
    plt.savefig('/data/twitter/sm-scraps-data/misc/figures/with_friend.png')
    plt.close()

    pie_data = [item for item in df.filter(F.col("text_word") != "friend").select("text_word", "count").head(10)]
    y = np.array([item[1] for item in pie_data])
    labes2 = [item[0] for item in pie_data]

    plt.pie(y, labels=labes2)
    plt.savefig('/data/twitter/sm-scraps-data/misc/figures/without_friend.png')
    plt.close()
