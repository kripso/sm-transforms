from utils.spark_logger import SparkLogger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import types as T
from PIL import Image
import unicodedata
import yaml
import json
import os

LOGGER = SparkLogger("file_handling")


def dataframe_to_html(df: DataFrame, name: str, path: str = './misc/html_outs', limit_rows: int = 100) -> None:
    html = df.limit(limit_rows).toPandas().to_html()
    file_path = os.path.join(check_directory(path), f'{name}.html',)

    with open(file_path, "w") as f:
        f.write(html)

    print(f'https://vscode.kripso-world.com/proxy/5500/{file_path.replace("./misc/", "")}')


def load_dictionary(name: str, keys: list):
    with open(f"./conf/dictionary/{name}.yaml") as f:
        data = yaml.safe_load(f)
    return {key: data.get(key, {}) for key in keys}


def load_spark_config(config_string: str = 'DEFAULT_SPARK_CONF'):
    conf = []
    locations = ["./conf/spark_config.yaml", "./conf/spark_secret_config.yaml"]
    for location in locations:
        with open(location) as f:
            data = yaml.safe_load(f)
            conf.extend(data.get(config_string, []))

    return conf


def make_spark_events_dir():
    os.makedirs('/tmp/spark-events', exist_ok=True)


def load_asyncio_client_config():
    with open("./conf/config.yaml") as f:
        data = yaml.safe_load(f)
    return data.get("ASYNCIO_CLIENT_CONFIG", {})


def return_dir_path(path: str):
    return os.path.dirname(os.path.realpath(path))


def remove_accent(text: str) -> str:
    """
    Remove accent from text
    :param text: text to remove accent from
    :return: text without accent
    """
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8", "ignore")


def get_json_entries() -> dict:
    with open("/data/twitter/sm-scraps-data/datasets/raw/targets.nested.json", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            yield data


def save_image(img_name: str, image: Image) -> None:
    directory = check_directory(f"/data/twitter/sm-scraps-data/images/{remove_accent(img_name)}")
    index = len(os.listdir(directory))

    with open(f"{directory}/image_{index}.png", "wb") as f:
        image.save(f, "JPEG")


def check_directory(directory: str) -> str:
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def write_schema(df: DataFrame, path="/data/twitter/sm-scraps-data/schemas/", name="schema.json") -> None:
    with open(os.path.join(path, name), "w") as f:
        json.dump(df.schema.jsonValue(), f)


def load_schema(path="/data/twitter/sm-scraps-data/schemas/", name="schema.json") -> T.StructType():
    with open(os.path.join(path, name)) as f:
        schema = T.StructType().fromJson(json.load(f))
    return schema
