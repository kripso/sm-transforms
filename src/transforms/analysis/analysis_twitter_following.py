import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import configure, transform, Input, Output
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from utils.spark_logger import SparkLogger

logger = SparkLogger("Following_Analysis")


@configure()
@transform(
    out_failed_df=Output("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts.analysis/failed", build_datetime=False),
    out_successful_df=Output("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts.analysis/successful", build_datetime=False),
    in_df=Input("/data/twitter/sm-scraps-data/datasets/clean/twitter_accounts.following/all", merge_schema=True),
)
def compute(ctx, out_failed_df: Output, out_successful_df: Output, in_df: Input):

    # in_df: DataFrame = in_df.read_df()
    # failed_df = (
    #     in_df
    #     .filter(F.col('error').isNotNull())
    #     .filter(~F.col('error').startswith('401 Unauthorized'))
    #     .filter(~F.col('error').startswith('403 Forbidden'))
    #     .dropDuplicates(['followers_id'])
    # )

    # failed_df.show()
    # failed_df.summary().show()
    # out_failed_df.write_df(failed_df)

    successful_in: DataFrame = in_df.read_df()
    successful_df = successful_in.filter(F.col("error").isNull())

    successful_df.show()
    print(successful_df.count())
    # successful_df.summary().show()
    # out_successful_df.write_df(successful_df)
