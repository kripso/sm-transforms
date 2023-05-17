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
    Output("/data/twitter/sm-scraps-data/datasets/clean/extracted_entities_relations/analysis"),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/extracted_entities_relations/all"),
)
def compute(ctx, df_in: DataFrame):
    # df_in.summary().show()
    df_relations = (
        df_in
        .groupBy('relation')
        .count()
        # .agg(
        #     F.count('text').alias('distinct_count')
        # )
    )

    df_relations.show(100, truncate=False)
    df_entities = (
        df_in
        .withColumn('entities', F.explode(F.col('entities')))
        .groupBy('entities.type')
        .count()
        # .agg(
        #     F.count('text').alias('distinct_count')
        # )
    )

    df_entities.show(100, truncate=False)
    # return df
