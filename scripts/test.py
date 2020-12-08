# Databricks notebook source
# MAGIC %sh wget https://github.com/RafiKurlansik/project_test/blob/main/dist/my_package-0.1-py3-none-any.whl?raw=true

# COMMAND ----------

 dbutils.fs.mv("file:/databricks/driver/my_package-0.1-py3-none-any.whl?raw=true", 
               "dbfs:/dbacademy/joshuacook/my_package-0.1-py3-none-any.whl")

# COMMAND ----------

# MAGIC %pip install /dbfs/dbacademy/joshuacook/my_package-0.1-py3-none-any.whl

# COMMAND ----------

from my_package import math

math.squared(4)