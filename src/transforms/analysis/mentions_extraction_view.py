import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from utils.pyspark_utils import configure, transform, Input, Output
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from utils.spark_logger import SparkLogger
from utils.file_handling import dataframe_to_html
import numpy as np
import matplotlib.pyplot as plt
import json
import re

logger = SparkLogger("metadata_extraction_tweets")


@configure()
@transform(
    out_=Output("/data/twitter/sm-scraps-data/datasets/clean/tweets_mentions_analysis"),
    in_mentions=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets_mentions/mentions", merge_schema=True),
    in__mentions_count=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets_mentions/mentions_count", merge_schema=True),
)
def compute(ctx, out_: Output, in_mentions: Input, in__mentions_count: Input):
    in_df = in_mentions.read_df()
    out_.write_to_html(in_df, 'in_mentions')

    in_df = in__mentions_count.read_df()
    out_.write_to_html(in_df, 'in_mentions_count')
