import sys
import pandas as pd
import numpy as np
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from evidently import Dataset, DataDefinition, BinaryClassification, Report
from evidently.presets import ClassificationPreset

# Airflow uses an ephemeral task runner, so we push metrics to a Pushgateway registry
registry = CollectorRegistry()
F1_GAUGE = Gauge('model_f1_score', 'F1 Score from Evidently', registry=registry)
RECALL_GAUGE = Gauge('model_recall_score', 'Recall from Evidently', registry=registry)

def extract_metric(data, key):
    if isinstance(data, dict):
        if key in data and isinstance(data[key], (int, float)):
            return data[key]
        for v in data.values():
            result = extract_metric(v, key)
            if result is not None:
                return result
    elif isinstance(data, list):
        for item in data:
            result = extract_metric(item, key)
            if result is not None:
                return result
    return None

def generate_dummy_data():
    # Simulate a stable evaluation dataset batch
    targets = np.random.choice([0, 1], 100, p=[0.7, 0.3])
    predictions = [t if np.random.rand() > 0.15 else 1 - t for t in targets]
    
    df = pd.DataFrame({
        "target": targets,
        "prediction": predictions
    })
    return df

def run_evidently_and_push():
    print("-" * 40)
    print("⏳ Calculating Evidently metrics...")
    df = generate_dummy_data()
    
    data_def = DataDefinition(
        classification=[BinaryClassification(target="target", prediction_labels="prediction")]
    )
    dataset = Dataset.from_pandas(df, data_definition=data_def)
    
    report = Report(metrics=[ClassificationPreset()])
    my_eval = report.run(reference_data=None, current_data=dataset)
    
    try:
        results = my_eval.dict()
    except Exception as e:
        print(f"⚠️ Erreur d'extraction : {e}")
        results = {}
    
    f1_score = extract_metric(results, 'f1') or 0.0
    recall_score = extract_metric(results, 'recall') or 0.0
    
    F1_GAUGE.set(f1_score)
    RECALL_GAUGE.set(recall_score)
    
    try:
        push_to_gateway('pushgateway:9091', job='evidently_monitoring_job', registry=registry)
        print(f"✅ Metric system pushed: F1={f1_score:.3f} | Recall={recall_score:.3f}")
    except Exception as e:
        print(f"❌ Failed to push to gateway: {e}")

if __name__ == '__main__':
    run_evidently_and_push()