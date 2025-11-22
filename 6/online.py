import pandas as pd
from river import compose, linear_model, preprocessing, metrics, stream
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import time
import numpy as np

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Hospital Readmission Strategy Comparison")

MODEL_NAME = "HospitalReadmissionModel-Online"
DEPLOY_EVERY_N_EVENTS = 1000  # Деплоим новую модель каждые 1000 событий


class RiverWrapper(mlflow.pyfunc.PythonModel):
    """Обертка для сохранения River-модели в формате MLflow PyFunc"""

    def __init__(self, model_artifact):
        self.model = model_artifact["model"]
        self.accuracy_tracker = model_artifact["accuracy_tracker"]

    def predict(self, context, model_input):
        dict_records = model_input.to_dict(orient='records')
        predictions = []
        for features in dict_records:
            # River модели возвращают вероятности для классификации
            proba = self.model.predict_proba_one(features)
            prediction = 1 if proba.get(1, 0) > 0.5 else 0
            predictions.append(prediction)
        return predictions


def train_online_model():
    print("--- Запуск онлайн-обучения (Online Training) ---")
    client = MlflowClient()

    # --- 1. Создание онлайн-модели для бинарной классификации ---
    # Логистическая регрессия с стандартизацией фичей
    model = compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression()
    )
    accuracy_tracker = metrics.Accuracy()
    auc_tracker = metrics.ROCAUC()

    # --- 2. Симуляция потока данных пациентов ---
    # Загружаем данные из parquet файла
    try:
        data_stream_df = pd.read_parquet(
            '../3/monitoring/data/reference_data.parquet')
        print(f"Загружено {len(data_stream_df)} записей пациентов")
    except Exception as e:
        print(f"Ошибка загрузки данных: {e}")
        return

    # Отбираем только основные числовые фичи для онлайн-обучения
    # В реальной системе более сложные фичи приходили бы из онлайн-feature store
    features_to_use = [
        'age', 'cholesterol', 'bmi', 'medication_count', 'length_of_stay',
        'systolic_bp', 'diastolic_bp'
    ]

    # Преобразуем категориальные фичи в числовые
    data_stream_processed = data_stream_df.copy()
    binary_mapping = {'Yes': 1, 'No': 0}
    for col in ['diabetes', 'hypertension']:
        if col in data_stream_processed.columns:
            data_stream_processed[col] = data_stream_processed[col].map(
                binary_mapping).fillna(0)
            features_to_use.append(col)

    target_to_use = 'readmitted_30_days'

    print(f"Используемые фичи: {features_to_use}")
    print(f"Целевая переменная: {target_to_use}")
    print(
        f"Баланс классов: {data_stream_processed[target_to_use].value_counts(normalize=True).to_dict()}")

    print(f"Симуляция потока из {len(data_stream_processed)} событий...")

    with mlflow.start_run(run_name="Online Training Run") as run:
        mlflow.log_param("features_used", str(features_to_use))
        mlflow.log_param("deploy_every_n", DEPLOY_EVERY_N_EVENTS)
        
        auc = 0

        for i, row in data_stream_processed.iterrows():
            # Подготавливаем фичи
            features = {k: float(row[k]) for k in features_to_use if k in row}
            true_value = int(0 if row[target_to_use] == 'No' else 1)

            # --- 3. Предсказание, оценка и обучение ---
            # Предсказываем ДО обучения
            prediction_proba = model.predict_proba_one(features)
            prediction = 1 if prediction_proba.get(1, 0) > 0.5 else 0

            # Обновляем метрики
            accuracy_tracker.update(true_value, prediction)
            auc_tracker.update(true_value, prediction_proba)

            # Обучаем модель на новом примере
            model.learn_one(features, true_value)

            # Логируем метрики в MLflow (делаем реже, чтобы не спамить)
            if (i + 1) % 500 == 0:
                current_accuracy = accuracy_tracker.get()
                current_auc = auc_tracker.get()
                mlflow.log_metric("rolling_accuracy",
                                  current_accuracy, step=i+1)
                mlflow.log_metric("rolling_auc", current_auc, step=i+1)

                if (i + 1) % 2000 == 0:
                    print(
                        f"Событие #{i+1}: Accuracy={current_accuracy:.3f}, AUC={current_auc:.3f}")

            # --- 4. Периодический деплой ---
            if (i + 1) % DEPLOY_EVERY_N_EVENTS == 0:
                current_accuracy = accuracy_tracker.get()
                current_auc = auc_tracker.get()

                print(
                    f"\nСобытие #{i+1}: Деплой новой версии онлайн-модели...")
                print(
                    f"Текущие метрики: Accuracy={current_accuracy:.3f}, AUC={current_auc:.3f}")

                # Сохраняем модель и метрики
                model_artifact = {
                    "model": model,
                    "accuracy_tracker": accuracy_tracker,
                    "auc_tracker": auc_tracker
                }

                mlflow.pyfunc.log_model(
                    artifact_path=f"model_step_{i+1}",
                    python_model=RiverWrapper(model_artifact),
                    registered_model_name=MODEL_NAME,
                    input_example=pd.DataFrame(
                        [{k: 0.0 for k in features_to_use}])
                )

                # Логируем финальные метрики для этой версии
                mlflow.log_metrics({
                    f"version_accuracy_{i+1}": current_accuracy,
                    f"version_auc_{i+1}": current_auc
                })
                
                if (current_auc < auc):
                    break
                else:
                    auc = current_auc

                # Переводим в Staging
                try:
                    latest_version = client.get_latest_versions(
                        MODEL_NAME, stages=["None"])[0]
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=latest_version.version,
                        stage="Staging",
                        archive_existing_versions=True
                    )
                    print(
                        f"Модель версии {latest_version.version} переведена в 'Staging'")

                except Exception as e:
                    print(f"Ошибка при переводе модели в Staging: {e}")

        # Финальное логирование метрик
        final_accuracy = accuracy_tracker.get()
        final_auc = auc_tracker.get()
        mlflow.log_metrics({
            "final_accuracy": final_accuracy,
            "final_auc": final_auc
        })

        print(f"\nОнлайн-обучение завершено!")
        print(
            f"Финальные метрики: Accuracy={final_accuracy:.3f}, AUC={final_auc:.3f}")
        print(f"Обучено на {len(data_stream_processed)} примерах")


if __name__ == "__main__":
    train_online_model()
