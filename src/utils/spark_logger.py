from contextlib import redirect_stdout
from datetime import datetime
from typing import Any, List


class SparkLogger:
    def __init__(self, name):
        self.name = name
        self._clear_file()

    def _clear_file(self) -> None:
        open(f'./src/logs/{self.name}.log', 'w').close()

    def info(self, *data: List[Any], terminal_out=False) -> None:
        with open(f'./src/logs/{self.name}.log', 'a') as f:
            with redirect_stdout(f):
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), *data)

        if terminal_out:
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), data)
