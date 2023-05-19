import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))

from utils.pyspark_utils import configure, transform_df, Output
from pyspark.sql import types as T
import csv
from datetime import datetime

SCHEMA = T.StructType(
    [
        T.StructField("id", T.StringType()),
        T.StructField("schema", T.StringType()),
        T.StructField("name", T.StringType()),
        T.StructField("aliases", T.ArrayType(T.StringType())),
        T.StructField("birth_date", T.DateType()),
        T.StructField("countries", T.ArrayType(T.StringType())),
        T.StructField("addresses", T.ArrayType(T.StringType())),
        T.StructField("identifiers", T.ArrayType(T.StringType())),
        T.StructField("sanctions", T.ArrayType(T.StringType())),
        T.StructField("phones", T.ArrayType(T.StringType())),
        T.StructField("emails", T.ArrayType(T.StringType())),
        T.StructField("dataset", T.StringType()),
        T.StructField("first_seen", T.DateType()),
        T.StructField("last_seen", T.DateType()),
    ]
)


def simple_data_fix(path: str):
    rows = []
    with open(path, encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=",")
        for row in csv_reader:
            values = {
                "id": row["id"],
                "schema": row["schema"],
                "name": row["name"],
                "aliases": list(item for item in row["aliases"].split(";")),
                "birth_date": None,
                "countries": list(item for item in row["countries"].split(";")),
                "addresses": list(item for item in row["addresses"].split(";")),
                "identifiers": list(item for item in row["identifiers"].split(";")),
                "sanctions": list(item for item in row["sanctions"].split(";")),
                "phones": list(item for item in row["phones"].split(";")),
                "emails": list(item for item in row["emails"].split(";")),
                "dataset": row["dataset"],
                "first_seen": datetime.strptime(row["first_seen"], "%Y-%m-%d %H:%M:%S") if row["first_seen"] != "" else None,
                "last_seen": datetime.strptime(row["last_seen"], "%Y-%m-%d %H:%M:%S") if row["last_seen"] != "" else None,
            }
            longer_date = ""
            for date in row["birth_date"].split(";"):
                longer_date = date if len(date.split("-")) > len(longer_date.split("-")) else longer_date
            if longer_date != "":
                if len(longer_date.split("-")) == 3:
                    values["birth_date"] = datetime.strptime(longer_date, "%Y-%m-%d")
                if len(longer_date.split("-")) == 1:
                    values["birth_date"] = datetime.strptime(longer_date, "%Y")
            rows.append(values)

    return rows


@configure()
@transform_df(
    Output("/data/twitter/sm-scraps-data/datasets/clean/targets", build_datetime=False),
)
def compute(spark):

    data = simple_data_fix("/data/twitter/sm-scraps-data/datasets/raw/targets.simple.csv")

    df = spark.createDataFrame(data, SCHEMA)

    return df
