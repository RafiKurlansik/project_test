# Databricks notebook source
# MAGIC %md
# MAGIC # MLflow Sklearn Training Tutorial
# MAGIC 
# MAGIC **Overview**
# MAGIC * Trains a Sklearn model several times with different values for `max_depth` hyperparameter.
# MAGIC * Algorithm is DecisionTreeRegressor with wine quality dataset.
# MAGIC * Shows different ways to view runs:
# MAGIC   * [MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos) - returns list of [RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) objects
# MAGIC   * [MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs) - returns list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects
# MAGIC   * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs) - returns Pandas DataFrame
# MAGIC   * [Experiment data source](https://docs.databricks.com/applications/mlflow/tracking.html#analyze-mlflow-runs-using-dataframes) - returns Spark DataFrame
# MAGIC * Find the best run for the experiment using MlflowClient.search_runs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup

# COMMAND ----------

# MAGIC %run ./Common

# COMMAND ----------

import mlflow
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

experiment_id, experiment_name = init()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Delete any existing runs

# COMMAND ----------

delete_runs(experiment_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

data_path = get_data_path()
data_path

# COMMAND ----------

import pandas as pd
data = pd.read_csv(data_path)
display(data)

# COMMAND ----------

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.30, random_state=42)

# The predicted column is col_label which is a scalar from [3, 9]
train_x = train.drop([col_label], axis=1)
test_x = test.drop([col_label], axis=1)
train_y = train[col_label]
test_y = test[col_label]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Pipeline

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn

# COMMAND ----------

def train(max_depth):
    with mlflow.start_run(run_name="sklearn") as run:
        run_id = run.info.run_uuid
        mlflow.log_param("max_depth", max_depth)
        mlflow.set_tag("version.mlflow", mlflow.__version__)
    
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(train_x, train_y)
        mlflow.sklearn.log_model(model, "sklearn-model")
        
        predictions = model.predict(test_x)
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        mlflow.log_metric("rmse", rmse)
        print(f"{rmse:5.3f} {max_depth:8d} {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train with different `max_depth` hyperparameter values

# COMMAND ----------

params = [1, 2, 4, 16]
print("RMSE  MaxDepth Run ID")
for p in params:
    train(p)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Show different ways to view the runs

# COMMAND ----------

# MAGIC %md
# MAGIC #### MlflowClient.list_run_infos
# MAGIC * [mlflow.tracking.MlflowClient.list_run_infos](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.list_run_infos)
# MAGIC * Returns a list of [RunInfo](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo) objects

# COMMAND ----------

infos = client.list_run_infos(experiment_id)
for info in infos:
    print(info.run_id, info.experiment_id, info.status)

# COMMAND ----------

# MAGIC %md
# MAGIC #### MLflowClient.search_runs
# MAGIC * [mlflow.tracking.MlflowClient.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.tracking.html#mlflow.tracking.MlflowClient.search_runs)
# MAGIC * Returns a list of [Run](https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.Run) objects,
# MAGIC * Allows for paging when you have a very large number of runs.
# MAGIC * Sorted by best metrics `rmse`.

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"])
for run in runs:
    print(run.info.run_id, run.data.metrics["rmse"], run.data.params)

# COMMAND ----------

# MAGIC %md
# MAGIC #### mlflow.search_runs
# MAGIC * [mlflow.search_runs](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.search_runs)
# MAGIC * Returns a Pandas dataframe.
# MAGIC * All `data` attributes are exploded into one flat column name space.
# MAGIC * Sorted by best metrics `rmse`.

# COMMAND ----------

runs = mlflow.search_runs(experiment_id)
runs = runs.sort_values(by=['metrics.rmse'])
runs

# COMMAND ----------

runs[["run_id","metrics.rmse","params.max_depth"]]

# COMMAND ----------

# MAGIC %md
# MAGIC #### Experiment data source
# MAGIC * Returns a Spark dataframe of all runs.
# MAGIC * Run `data` elements such as `params`, `metrics` and `tags` are nested.
# MAGIC * Background Documentation:
# MAGIC   * Databricks documentation:
# MAGIC     * [MLflow Experiment Data Source](https://docs.databricks.com/data/data-sources/mlflow-experiment.html#mlflow-exp-datasource)
# MAGIC     * [Analyze MLflow runs using DataFrames
# MAGIC ](https://docs.databricks.com/applications/mlflow/tracking.html#analyze-mlflow-runs-using-dataframes)
# MAGIC   * [Analyzing Your MLflow Data with DataFrames](https://databricks.com/blog/2019/10/03/analyzing-your-mlflow-data-with-dataframes.html) - blog - 2019-10-03

# COMMAND ----------

from pyspark.sql.functions import *
df_runs = spark.read.format("mlflow-experiment").load(experiment_id)
df_runs.createOrReplaceTempView("runs")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Query with Spark DataFrame API

# COMMAND ----------

df_runs = df_runs.sort(asc("metrics.rmse"))
display(df_runs)

# COMMAND ----------

display(df_runs.select("run_id", round("metrics.rmse",3).alias("rmse"),"params"))

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Query as SQL

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc

# COMMAND ----------

# MAGIC %sql select run_id, metrics.rmse, params from runs order by metrics.rmse asc limit 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### Find the best run
# MAGIC 
# MAGIC * We use `MlflowClient.search_run` to find the best run

# COMMAND ----------

runs = client.search_runs(experiment_id, order_by=["metrics.rmse ASC"], max_results=1)
best_run = runs[0]
best_run

# COMMAND ----------

print("Run ID:",best_run.info.run_id)
print("RMSE:",best_run.data.metrics["rmse"])

# COMMAND ----------

display_run_uri(experiment_id, best_run.info.run_id)

# COMMAND ----------

