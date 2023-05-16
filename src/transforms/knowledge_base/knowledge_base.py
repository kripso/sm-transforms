import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

# spark imports
from utils.pyspark_utils import configure, transform_df, Input, Output
from utils.spark_logger import SparkLogger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

# ml imports
from utils.modeling_classes import JointNERAndREModel, JointNERAndREDataset
import utils.ml_utils as ml_utils
import wandb
import torch

LOGGER = SparkLogger("Applied Model")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABELS_TO_IDS, IDS_TO_LABELS = ml_utils.load_labels()
RELATIONS_TO_IDS, IDS_TO_RELATIONS = ml_utils.load_relations()

SCHEMA = T.StructType([
    T.StructField('user_name', T.StringType()),
    T.StructField('text', T.StringType()),
    T.StructField('relation', T.StringType()),
    T.StructField('entities', T.ArrayType(T.StructType([
        T.StructField('type', T.StringType()),
        T.StructField('location', T.LongType()),
    ]))),
])


def apply_model(partition, model):
    results = []

    model.eval()
    with torch.inference_mode():
        for row in partition:
            encoded = JointNERAndREDataset.tokenize(row['text'], is_split=False, return_tensors='pt').to(DEVICE)
            model_out = model(encoded["input_ids"], attention_mask=encoded["attention_mask"])
            ner_predictions = torch.argmax(model_out.ner_probs.view(-1, model.num_labels), axis=1).tolist()
            re_predictions = torch.argmax(model_out.re_probs, axis=1).tolist()[0]

            metadata = {"relation": IDS_TO_RELATIONS.get(re_predictions), 'entities': []}
            index = 0
            for token, mapping in zip(ner_predictions, encoded["offset_mapping"].view(-1, 2).tolist()):
                if mapping[0] == 0 and mapping[1] != 0:
                    metadata['entities'].append({'type': IDS_TO_LABELS.get(token), 'location': index})
                    index += 1

            results.append({'user_name': row['user_name'], 'text': row['text'], **metadata})

    return results


@configure(master="local[12]")
@transform_df(
    Output("/data/twitter/sm-scraps-data/datasets/clean/extracted_entities_relations/all"),
    df_in=Input("/data/twitter/sm-scraps-data/datasets/clean/tweets/all"),
)
def compute(ctx, df_in: DataFrame):
    wandb.init(project='dp-project-validate', config=None, dir=CURRENT_DIR)
    model = JointNERAndREModel().to(DEVICE)
    model = ml_utils.load_model(model, model_version='v128')
    wandb.finish()

    df = (
        df_in
        .select(
            'user_name',
            'text',
        )
        .dropna(subset='text')
    )

    df = (
        df
        .repartition(1024)
        .rdd
        .mapPartitions(lambda partition: apply_model(partition, model))
        .toDF(SCHEMA)
    )

    return df
