"""Console script for forecast_dataset_tools."""
from . import logging_config  # isort:skip

import logging
import shutil

import click

from .meteogram_extract import process_file

_log = logging.getLogger(__name__)


context_settings = {"max_content_width": shutil.get_terminal_size().columns}


@click.command(context_settings=context_settings)
@logging_config.log_level_option
@click.argument(
    "meteogram_img",
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False),
)
def cli(log_level, meteogram_img):
    """
    Digitize the data found on METEOGRAM_IMG files and save it to a csv file in
    the same directory, with the same file name but .csv extension.
    """
    logging.getLogger("meteogram_extract").setLevel(log_level)

    files_with_errors = []

    for fname in meteogram_img:
        try:
            df = process_file(fname, write_csv=True, plot=False)
        except SystemExit:
            break
        except Exception as e:
            _log.error(f"Failed to process {fname}.", exc_info=True)
            files_with_errors.append(fname)

    if files_with_errors:
        _log.info(f"Failures to process the following files:")
        for fname in files_with_errors:
            _log.info(fname)


if __name__ == "__main__":
    cli()
