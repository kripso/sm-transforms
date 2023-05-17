import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

# spark imports
from utils.pyspark_utils import configure, transform_df, Input, Output
from utils.spark_logger import SparkLogger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

LOGGER = SparkLogger("Applied Model")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@configure(master="local[*]")
@transform_df(
    Output("/data/twitter/sm-scraps-data/datasets/clean/extracted_entities_relations/visualisation"),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/extracted_entities_relations/all"),
)
def compute(ctx, df_in: DataFrame):
    # df_in.summary().show()
    df = (
        df_in
        .select(
            F.col('user_name').alias('source'),
            F.col('text').alias('target'),
        )
    )

    df.show(100, truncate=False)

    df = (
        df_in
        .select(
            F.col('text').alias('target'),
            F.explode(F.col('entities')).alias('entities'),
            F.col('relation'),
        )
        .select(
            "*",
            'entities.*',
        )
        .drop('entities')
    )

    df.show(100, truncate=False)
