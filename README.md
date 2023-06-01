# Meteogram Extract

Digitize the data from meteogram forecast images from the Turkish Meteorology
Department (MGM). A command-line interface is provided to facilitate use.

Meteograms are available from the MGM website at
https://mgm.gov.tr/tahmin/wrf-meteogram.aspx.

## Installation

Currently this package is only distributed via its GitHub repository and is not
on PyPI.

To install using pip:

```
pip install git+https://github.com/pdb5627/meteogram_extract/
```

Dependencies are listed in `environment.yml`, which can be used to install the
dependencies in a conda environment using conda. (Add the `--name` option to
specify a different name than the one in the environment file.)

For a new environment:
```
conda env create -f environment.yml
```

To add dependencies to an existing environment:
```
conda activate myenv
conda env update --file environment.yml  --name myenv
```

## CLI usage

When installed using pip, a new script named `meteogram_extract` is installed.
It's usage can be seen by running it with the `--help` option or as follows:
```
Usage: meteogram_extract [OPTIONS] [METEOGRAM_IMG]...

  Digitize the data found on METEOGRAM_IMG files and save it to a csv file in the
  same directory, with the same file name but .csv extension.

Options:
  -l, --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]
                                  Logging level to use for this package  [default: INFO]
  --help                          Show this message and exit.
```

## MGM server issues with SSL

The MGM meteograms can be downloaded using a tool such as curl from URLs of the
form `https://www.mgm.gov.tr//FTPDATA/sht/wrf/<il>_<ilçe>.png`. (MGM may change
this at any time.)

As of May 2023, the MGM server is not able to negotiate an ssl connection that
openssl's default settings will allow. This can be worked around by using an
alternative openssl.cnf configuration, or by using an older version of openssl
(v1.1.1). A working openssl.cnf is available at
https://github.com/pdb5627/meteogram_extract/blob/master/openssl.cnf. It
can be used while running a script by setting the OPENSSL_CONF envionment
variable to its location on disk. On a Linux shell, the command would look like
this:

```
>>> OPENSSL_CONF=~/openssl.cnf curl ...
```
