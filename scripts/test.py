# Databricks notebook source
# MAGIC %pip install git+https://github.com/RafiKurlansik/project_test

# COMMAND ----------

from my_package import math

math.squared(4)

# COMMAND ----------

# MAGIC %md ### Install with Token

# COMMAND ----------

token = dbutils.secrets.get(scope = <USERNAME>, key = <KEY>)

%pip install git+https://$token@github.com/RafiKurlansik/project_test.git