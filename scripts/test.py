# Databricks notebook source
# MAGIC %pip install git+https://github.com/RafiKurlansik/project_test

# COMMAND ----------

from my_package import math

math.squared(4)

# COMMAND ----------

# MAGIC %md ### Install with Token

# COMMAND ----------

# MAGIC token = dbutils.secrets.get(scope = <USERNAME>, key = <KEY>)
# MAGIC
# MAGIC %pip install git+https://$token@github.com/RafiKurlansik/project_test.git

# COMMAND ----------

# MAGIC %md ### Install via Download

# COMMAND ----------

# MAGIC %sh wget https://github.com/RafiKurlansik/project_test/blob/main/dist/my_package-0.1-py3-none-any.whl?raw=true

# COMMAND ----------

 dbutils.fs.mv("file:/databricks/driver/my_package-0.1-py3-none-any.whl?raw=true",
               "dbfs:/dist/my_package-0.1-py3-none-any.whl")

# COMMAND ----------

# MAGIC %pip install /dist/joshuacook/my_package-0.1-py3-none-any.whl
