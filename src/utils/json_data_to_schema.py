from pyspark.sql import types as T
from typing import Union
import json
import re

data = '{"roles": {"role_name": {"id": "enrollment:resource-management-administrator","displayName": "Resource management administrator","description": "Manage resource allocation and monitor usage on the platform","template": {"type": "tracking","tracking": {"templateId": "enrollment:resource-management-administrator"}},"canAssigns": ["enrollment:resource-management-viewer"],"operations": ["resource-management:usage-account:edit","foundry-oql:manage-database","streaming-profiles:manage","resource-management:resource-queue:create-within-limits","control-panel:enrollment:discover","resource-management:usage-account:create","enrollment:resource-management-viewer","spark-profiles:import","resource-management:usage-account:edit-assignments","resource-management:currency:view","resource-management:budgets:view","resource-management:resource-queue:edit-assignments","internal-tables:export-rm-data-for-enrollment","control-panel:vector:manage-wmq","resource-management:budgets:edit","resource-management:resource-queue:edit-within-limits"],"trackedWorkflows": ["export-rm-enrollment-workflow","resource-management-currency-workflow","manage-streaming-profile-imports-workflow","foundry-oql-admin-workflow","manage-spark-profiles-imports-workflow","resource-queues-admin-workflow","discover-enrollment-workflow","usage-accounts-admin-workflow","manage-code-workbook-warm-module-queues-workflow"],"changeSequenceNum": 0,"isDeleted": false}}}'  # noqa


def return_instance(obj, key=None) -> T.DataType:
    if isinstance(obj, int):
        if obj > 2147483647:
            return T.StructField(f"{key}", T.LongType()), T.LongType()
        return T.StructField(f"{key}", T.IntegerType()), T.IntegerType()
    if isinstance(obj, float):
        return T.StructField(f"{key}", T.DoubleType()), T.DoubleType()
    if isinstance(obj, str):
        re_result = re.search(r"(\d{4}-[01]\d-[0-3]\d)(T[0-2]\d:[0-5]\d:[0-5]\d(?:\.\d+)?Z?)?", obj)
        if re_result is not None:
            if re_result.groups()[1] is not None:
                return T.StructField(f"{key}", T.TimestampType()), T.TimestampType()
            return T.StructField(f"{key}", T.DateType()), T.DateType()
        return T.StructField(f"{key}", T.StringType()), T.StringType()
    return T.StructField(f"{key}", T.DataType()), T.DataType()


def get_schema(index, _obj, key=None):
    if isinstance(_obj, dict):
        T_type = T.StructType([get_schema(index + 1, _obj[_key], _key) for _key in _obj.keys()])
        return T_type if index == 0 else T.StructField(f"{key}", T_type)

    if isinstance(_obj, list):
        T_types = set()
        for i, tmp_obj in enumerate(_obj):
            if isinstance(tmp_obj, Union[list, dict]):
                if len(_obj) == 1:
                    T_types.add(get_schema(0, tmp_obj, key))
                else:
                    T_types.add(T.StructField(f"type_{i}", get_schema(0, tmp_obj, key)))
            else:
                T_types.add(return_instance(tmp_obj, key)[1])
        if len(T_types) == 1:
            return T.StructField(f"{key}", T.ArrayType(T_types.pop()))
        if len(T_types) > 1:
            return T.StructField(f"{key}", T.ArrayType(T.StructType(list(T_types))))

    return return_instance(_obj, key)[0]


if __name__ == "__main__":
    _json = json.loads(data)
    print(get_schema(0, _json))
