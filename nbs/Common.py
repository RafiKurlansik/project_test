# Databricks notebook source
# Databricks notebook source

# COMMAND ----------

def _get_notebook_tag(tag):
    tag = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get(tag)
    return None if tag.isEmpty() else tag.get()

# COMMAND ----------

import os
import platform
import mlflow
import mlflow.spark
import pyspark
print("MLflow Version:", mlflow.__version__)
print("Spark Version:", spark.version)
print("PySpark Version:", pyspark.__version__)
print("sparkVersion:", _get_notebook_tag("sparkVersion"))
print("DATABRICKS_RUNTIME_VERSION:", os.environ.get('DATABRICKS_RUNTIME_VERSION',None))
print("Python Version:", platform.python_version())
client = mlflow.tracking.MlflowClient()

# COMMAND ----------

def init():
    experiment_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    print("Experiment name:",experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)
    print("Experiment ID:",experiment.experiment_id)
    return experiment.experiment_id, experiment_name

# COMMAND ----------

def delete_runs(experiment_id):
    run_infos = client.list_run_infos(experiment_id)
    print(f"Found {len(run_infos)} runs for experiment_id {experiment_id}")
    for run_info in run_infos:
        client.delete_run(run_info.run_id)

# COMMAND ----------

data_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"

def get_data_path():
    return data_path

# COMMAND ----------

col_label = "quality"
col_prediction = "prediction"

# COMMAND ----------

host_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get("browserHostName").get()

# COMMAND ----------

def display_run_uri(experiment_id, run_id):
    uri = f"https://{host_name}/#mlflow/experiments/{experiment_id}/runs/{run_id}"
    displayHTML("""<b>Run URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_experiment_uri(experiment_id):
    uri = "https://{}/#mlflow/experiments/{}".format(host_name, experiment_id)
    displayHTML("""<b>Experiment URI:</b> <a href="{}">{}</a>""".format(uri,uri))

# COMMAND ----------

def display_registered_model_uri(model_name):
    uri = f"https://{host_name}/#mlflow/models/{model_name}"
    displayHTML("""<b>Registered Model URI:</b> <a href="{}">{}</a>""".format(uri,uri))