import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import configure, transform, Input, Output, DataType, SaveMode
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from utils.spark_logger import SparkLogger
from pyspark.sql import SparkSession


logger = SparkLogger("Following_Analysis")


@configure('MINIO_SPARK_CONF')
@transform(
    out_df=Output("s3a://test/tweets_1_000_000", build_datetime=False, data_type=DataType.JSON, save_mode=SaveMode.OVERWRITE),
    in_df=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets/all", merge_schema=True),
)
def compute(ctx: SparkSession, in_df: Input, out_df: Output):

    in_df = in_df.read_df()
    # print(in_df.count())
    out_df.write_df(in_df.limit(1_000_000).coalesce(1))
