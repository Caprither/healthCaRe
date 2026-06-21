from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
}

with DAG(
    '4_retrain_model',
    default_args=default_args,
    description='Executes the model training script',
    schedule_interval=None,  # 'None' means it only runs when triggered manually or by another DAG
    start_date=datetime(2026, 6, 21),
    catchup=False
) as dag:

    # This task runs your existing train.py script
    run_training = BashOperator(
        task_id='execute_train_script',
        bash_command='python /opt/airflow/src/models/train.py'
    )