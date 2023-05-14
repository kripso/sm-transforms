from functools import wraps
from typing import Callable, Concatenate, Dict, List, Optional, ParamSpec, TypeVar, Union, Iterable
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark import SparkConf
from datetime import datetime
from enum import Enum
import time
from utils.file_handling import load_spark_config, return_dir_path, dataframe_to_html, make_spark_events_dir
from utils.spark_logger import SparkLogger
from concurrent.futures import ThreadPoolExecutor
from utils.AsyncioClient import AsyncioClient
import asyncio
import more_itertools

LOGGER = SparkLogger("Utils")
DEFAULT_SPARK_CONF = load_spark_config()
DIR = return_dir_path(__file__)

P = ParamSpec("P")
A = TypeVar("A")
B = TypeVar("B")


#
# Enums
#
class DataType(str, Enum):
    CSV = ".csv"
    PARQUET = ".parquet"
    JSON = ".json"
    NONE = None

    def __str__(self):
        return self.value


class SaveMode(str, Enum):
    OVERWRITE = "overwrite"
    APPEND = "append"
    IGNORE = "ignore"

    def __str__(self):
        return self.value


#
# Classes
#
class Input:
    def __init__(
        self, path: str, data_type=DataType.PARQUET, delimeter=",", schema: Optional[T.StructType] = None, merge_schema=True, spark_session=None
    ) -> "Input":
        self.merge_schema = merge_schema
        self.data_type = data_type
        self.delimeter = delimeter
        self.spark = spark_session
        self.schema = schema
        self.path = path

    def get_path(self) -> str:
        return self.path

    def add_spark_session(self, spark_session: SparkSession) -> None:
        self.spark = spark_session

    def read_df(self) -> DataFrame:
        if DataType.CSV == self.data_type:
            df = self.spark.read.csv(self.path, inferSchema=True, header=True, delimiter=self.delimeter)
            df = df if self.schema is None else self.spark.createDataFrame(data=df.rdd, schema=self.schema)
        if DataType.PARQUET == self.data_type:
            df = self.spark.read.parquet(self.path, mergeSchema=self.merge_schema)
        if DataType.JSON == self.data_type:
            df = self.spark.read.json(self.path, schema=self.schema)
        return df


class Output(Input):
    def __init__(
        self, path: str, partition_by: Optional[str | List[str]] = None, build_datetime=True, save_mode=SaveMode.APPEND, **kwargs
    ) -> "Output":
        super().__init__(path, **kwargs)
        self.build_datetime = build_datetime
        self.partition_by = partition_by
        self.save_mode = save_mode

    def _datetime_now(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _new_partition_keys(self, keys: Optional[str | List[str]], add_key: str) -> str | List[str]:
        if keys is None:
            return add_key
        if isinstance(keys, str):
            return list({keys, add_key})
        return list({*keys, add_key})

    def write_to_html(self, df: DataFrame, name: str):
        return dataframe_to_html(df, f'{self.path.split("/")[-1]}_{name}')

    def write_df(self, df: DataFrame) -> None:
        if self.build_datetime:
            df = df.withColumn("build_datetime", F.lit(self._datetime_now()).cast(T.TimestampType()))
            self.partition_by = self._new_partition_keys(self.partition_by, "build_datetime")

        if DataType.CSV == self.data_type:
            df.write.mode(self.save_mode).csv(self.path, header=True)
        if DataType.PARQUET == self.data_type:
            df.write.mode(self.save_mode).parquet(self.path, partitionBy=self.partition_by)
        if DataType.JSON == self.data_type:
            df.write.mode(self.save_mode).json(self.path)

    def read_df(self) -> DataFrame:
        try:
            return super().read_df()
        except BaseException:
            return self.spark.createDataFrame(data=self.spark.sparkContext.emptyRDD(), schema=T.StructType([]))


#
# Decorators
#
def path_check(**dataframes: Dict[str, Input | Output]) -> None:
    paths = {}
    for key in dataframes:
        _df_type = "Input" if type(dataframes[key]) is Input else "Output"
        _path = dataframes[key].get_path()
        if _path in paths:
            if _df_type == paths[_path]:
                raise BaseException(f"Multiple Inputs Collision: {_path}")
            else:
                raise BaseException(f"Cicling Dependency Error: {_path}")

        paths[_path] = _df_type


def path_check_2(**dataframes: Dict[str, Input | Output]) -> None:
    _paths = set()
    for key in dataframes:
        _path = dataframes[key].get_path()
        if _path in _paths:
            raise BaseException(f"Dependency Error On Path: {_path}")

        _paths.add(_path)


# for use with docker set -> master = "spark://spark-master:7077"
def configure(conf="DEFAULT_SPARK_CONF", master="local[*]", auto=True):
    def configure_func(func):
        @wraps(func)
        def wrapped() -> SparkSession:
            make_spark_events_dir()

            spark_config = load_spark_config(conf)
            spark_config.extend(list(item for item in DEFAULT_SPARK_CONF if item[0] not in set(item[0] for item in conf)))

            spark_session = (
                SparkSession.builder.config(conf=(SparkConf().setAppName("My-Spark-Application").setMaster(master).setAll(spark_config)))
            ).getOrCreate()

            spark_session.sparkContext.addFile(DIR, recursive=True)
            return func(spark_session)
        if auto:
            return wrapped()
        return wrapped
    return configure_func


def transform(**dataframes: Dict[str, Input | Output]):
    def transform_func(func: Callable[[Optional[SparkSession], Dict[str, Input | Output]], None]):
        @wraps(func)
        def wrapped(spark_session: Optional[SparkSession] = None):
            path_check(**dataframes)
            for key in dataframes:
                dataframes[key].add_spark_session(spark_session)

            try:
                func(spark_session, **dataframes)
            except TypeError:
                func(**dataframes)

        return wrapped

    return transform_func


def transform_df(output: Output, **dataframes: Dict[str, Input]):
    def transform_df_func(func: Callable[[Optional[SparkSession], Dict[str, DataFrame]], Optional[DataFrame]]):
        @wraps(func)
        def wrapped(spark_session: Optional[SparkSession] = None):
            path_check(**(dataframes | {"_": output}))

            for key in dataframes:
                dataframes[key].add_spark_session(spark_session)
                dataframes[key] = dataframes[key].read_df()

            try:
                dataframe = func(spark_session, **dataframes)
            except TypeError:
                dataframe = func(**dataframes)
            finally:
                if dataframe is not None:
                    output.write_df(dataframe)

        return wrapped

    return transform_df_func


def limit_rate(seconds: Optional[int] = None, limit: Optional[int] = None, interval_in_secods: Optional[int] = None, grace_period=0.2):
    def limit_rate_func(func: Callable[P, B]):
        @wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> B:
            _start_time = time.time()
            result = func(*args, **kwargs)
            _end_time = time.time()

            if seconds is not None:
                _sleep_time = seconds - (_end_time - _start_time)
            else:
                _sleep_time = (interval_in_secods / limit) - (_end_time - _start_time)

            if _sleep_time > 0.0:
                time.sleep(_sleep_time + grace_period)

            return result

        return wrapped

    return limit_rate_func


def threaded(max_workers: int, flatten: Optional[bool] = False):
    def threaded_func(func: Callable[Concatenate[A, P], B]):
        @wraps(func)
        def wrapper(partition: Iterable[A], *args: P.args, **kwargs: P.kwargs) -> Union[B, List[B]]:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(func, row, *args, **kwargs) for row in partition}
            results = [_future.result() for _future in futures]
            if flatten:
                return more_itertools.flatten(results)
            return results

        return wrapper

    return threaded_func


def asynchronous(max_workers=5, flatten: Optional[bool] = False, return_exceptions=False):
    def asynchronous_func(func: Callable[Concatenate[A, P], B]):
        @wraps(func)
        def wrapper(partition: Iterable[A], *args: P.args, **kwargs: P.kwargs) -> Union[B, List[B]]:
            semaphore = asyncio.BoundedSemaphore(max_workers)

            async def _limited(row, *_args, **_kwargs):
                async with semaphore:
                    return await func(row, *_args, **_kwargs)

            async def _wrapped(_partition, *_args, **_kwargs):
                tasks = [asyncio.ensure_future(_limited(row, *_args, **_kwargs)) for row in _partition]
                results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
                if flatten:
                    return more_itertools.flatten(results)
                return results

            return asyncio.run(_wrapped(partition, *args, **kwargs))

        return wrapper

    return asynchronous_func


def aiohttp_request(max_workers=5, flatten: Optional[bool] = False, return_exceptions=False):
    def aiohttp_request_func(func: Callable[Concatenate[A, AsyncioClient, P], B]):
        @wraps(func)
        def wrapper(partition: Iterable[A], client: AsyncioClient, *args: P.args, **kwargs: P.kwargs) -> Union[B, List[B]]:
            async def _wrapped_client(_partition, _client, *_args, **_kwargs):
                tasks = [func(row, _client, *_args, **_kwargs) for row in _partition]
                results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
                if flatten:
                    return more_itertools.flatten(results)
                return results

            return asyncio.run(client.with_session(max_workers, lambda: _wrapped_client(partition, client, *args, **kwargs)))

        return wrapper

    return aiohttp_request_func
