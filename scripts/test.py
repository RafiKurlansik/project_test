# Databricks notebook source

# MAGIC %pip install /dbfs/dbacademy/joshuacook/my_package-0.1-py3-none-any.whl

# COMMAND ----------

from my_package import math

math.squared(4)
