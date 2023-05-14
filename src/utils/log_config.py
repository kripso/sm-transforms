from logging import Logger
import logging.config
import yaml


class NoConsoleFilter(logging.Filter):
    def filter(self, record) -> bool:
        return record.levelname != "INFO"


def get_logger(name: str) -> Logger:
    with open("./conf/log_config.yaml", "r") as f:
        log_config = yaml.safe_load(f.read())
        log_config["handlers"]["file_handler"]["filename"] = f"./src/logs/{name}.log"
        log_config["filters"]["no_console_filter"]["()"] = NoConsoleFilter

        logging.config.dictConfig(log_config)

    return logging.getLogger(name)
