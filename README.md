# How This Repo Works

## `my_package`

A python package is built in the my_package directory.

## Build and Deploy Package

### Local Development

1. Make any changes to `my_package`
1. Build the wheel

   `python setup.py bdist_wheel`

1. Commit the built package to github

### On Databricks

This is all done in the file `scripts/test`.

1. use `wget` to pull the package onto the driver
1. use `dbutils.mv` it from driver to dbfs
1. pip install from dbfs
