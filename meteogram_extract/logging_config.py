"""
Logging configuration must be performed prior to the first logging message in order
for it to be effective and logging messages to be seen. This module should be imported
first in case other modules perform logging when they are imported. The logging levels
of different modules can be adjusted in the loggers section of the logging_config dict.
"""

import logging
import logging.config

logging_config = {
    "version": 1,
    "formatters": {
        "basic": {"format": "%(levelname)-8s: %(message)s"},
        "detailed": {"format": "%(asctime)s %(levelname)-8s %(name)s: %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "basic",
            "level": "DEBUG",
        }
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "WARNING"},
        "forecast_dataset_tools": {"level": "WARNING"},
    },
}

logging.config.dictConfig(logging_config)

import click

log_level_map = {
    logging.getLevelName(x): x
    for x in range(1, 101)
    if not logging.getLevelName(x).startswith("Level")
}

log_level_option = click.option(
    "-l",
    "--log-level/",
    "log_level",
    type=click.Choice(log_level_map.keys(), case_sensitive=False),
    default="INFO",
    show_default=True,
    help="Logging level to use for this package",
)
