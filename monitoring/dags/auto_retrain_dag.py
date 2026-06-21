from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime

# A mock function to simulate checking the latest F1 score from Prometheus/Evidently
def check_latest_f1_score():
    # In reality, this would query your Prometheus API for the latest 'model_f1_score'
    latest_f1 = 0.65  # Simulating a degraded model score
    
    # If the score is below our 0.70 threshold, trigger retraining
    if latest_f1 < 0.70:
        return 'trigger_retraining_pipeline'
    return 'model_is_healthy'

with DAG(
    '5_auto_retraining_controller',
    description='Evaluates metrics and triggers retraining if necessary',
    schedule_interval='@daily',
    start_date=datetime(2026, 6, 21),
    catchup=False
) as dag:

    # Step 1: Check the score
    evaluate_metrics = BranchPythonOperator(
        task_id='evaluate_model_degradation',
        python_callable=check_latest_f1_score
    )

    # Step 2a: The path to take if the model is failing
    trigger_retraining_pipeline = TriggerDagRunOperator(
        task_id='trigger_retraining_pipeline',
        trigger_dag_id='4_retrain_model',  # This exactly matches the DAG name in retrain_dag.py
        wait_for_completion=False
    )

    # Step 2b: The path to take if the model is fine
    model_is_healthy = EmptyOperator(
        task_id='model_is_healthy'
    )

    # Map out the workflow branches
    evaluate_metrics >> [trigger_retraining_pipeline, model_is_healthy]