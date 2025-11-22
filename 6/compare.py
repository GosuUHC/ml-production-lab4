# compare_strategies.py

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, roc_curve, precision_recall_curve, auc
import time
import numpy as np

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Hospital Readmission Strategy Comparison")

BATCH_MODEL_NAME = "HospitalReadmissionModel-Batch"
ONLINE_MODEL_NAME = "HospitalReadmissionModel-Online"


def compare_models():
    print("--- Сравнение пакетной и онлайн-стратегий ---")
    client = MlflowClient()

    # --- 1. Загрузка моделей ---
    try:
        batch_model_info = client.get_latest_versions(
            BATCH_MODEL_NAME, stages=["Staging"])[0]
        batch_model_uri = f"models:/{batch_model_info.name}/{batch_model_info.version}"
        batch_model = mlflow.pyfunc.load_model(batch_model_uri)
        print(f"Загружена Batch-модель: версия {batch_model_info.version}")

        online_model_info = client.get_latest_versions(
            ONLINE_MODEL_NAME, stages=["Staging"])[0]
        online_model_uri = f"models:/{online_model_info.name}/{online_model_info.version}"
        online_model = mlflow.pyfunc.load_model(online_model_uri)
        print(f"Загружена Online-модель: версия {online_model_info.version}")
    except IndexError as e:
        print(
            f"Ошибка: Не удалось загрузить одну из моделей. Убедитесь, что скрипты обучения были запущены. {e}")
        return

    # --- 2. Подготовка тестовых данных ---
    test_df = pd.read_parquet(
        '../2/hospital_readmissions/feature_repo/data/patient_features.parquet')

    categorical_columns = ['gender', 'diabetes', 'hypertension',
                           'discharge_destination', 'bmi_category', 'age_group']

    # Преобразуем бинарные категориальные фичи
    binary_mapping = {'Yes': 1, 'No': 0}
    for col in ['diabetes', 'hypertension']:
        if col in test_df.columns:
            test_df[col] = test_df[col].map(binary_mapping).fillna(0)

    # One-hot encoding для batch модели
    batch_df_encoded = pd.get_dummies(
        test_df, columns=categorical_columns, prefix=categorical_columns)

    # Убираем служебные колонки
    columns_to_drop = ['patient_id', 'event_timestamp',
                       'created_timestamp', 'readmitted_30_days']
    feature_columns = [col for col in batch_df_encoded.columns
                       if col not in columns_to_drop]

    X_test_batch = batch_df_encoded[feature_columns]

    # Фичи для онлайн-модели
    features_to_use = [
        'age', 'cholesterol', 'bmi', 'medication_count', 'length_of_stay',
        'systolic_bp', 'diastolic_bp', 'diabetes', 'hypertension'
    ]

    X_test_online = test_df[features_to_use].copy()
    target_to_use = 'readmitted_30_days'
    y_test = test_df[target_to_use]
    y_test_numeric = y_test.map(binary_mapping).fillna(0)

    # --- Исправление типов данных ---
    print("Исправление типов данных...")
    
    # Batch модель
    X_test_batch['age'] = X_test_batch['age'].astype('int64')
    numeric_columns_batch = ['cholesterol', 'medication_count', 'length_of_stay', 
                           'systolic_bp', 'diastolic_bp', 'high_bp_high_chol',
                           'multiple_conditions', 'age_medication_interaction',
                           'treatment_intensity', 'high_cholesterol', 'high_bmi', 'extended_stay']
    
    for col in numeric_columns_batch:
        if col in X_test_batch.columns:
            X_test_batch[col] = X_test_batch[col].astype('int64')
    
    # Online модель
    for col in features_to_use:
        if col in X_test_online.columns:
            X_test_online[col] = X_test_online[col].astype('float64')

    # --- 3. Оценка точности с вероятностями для AUC ---
    print("\nОценка точности моделей...")
    
    try:
        # Получаем предсказания
        batch_preds = batch_model.predict(X_test_batch)
        online_preds = online_model.predict(X_test_online)

        # Функция для получения вероятностей
        def get_probabilities(preds):
            if isinstance(preds[0], str):
                # Если модель возвращает 'Yes'/'No', создаем псевдо-вероятности
                return np.array([0.9 if pred == 'Yes' else 0.1 for pred in preds])
            elif hasattr(preds[0], '__len__') and len(preds[0]) > 1:
                # Если модель возвращает вероятности для нескольких классов
                return np.array([pred[1] if len(pred) > 1 else pred[0] for pred in preds])
            else:
                # Если модель возвращает числовые scores
                return np.array([float(pred) for pred in preds])

        # Получаем вероятности для положительного класса
        batch_probs = get_probabilities(batch_preds)
        online_probs = get_probabilities(online_preds)

        # Вычисляем AUC-ROC
        batch_auc = roc_auc_score(y_test_numeric, batch_probs)
        online_auc = roc_auc_score(y_test_numeric, online_probs)

        # Вычисляем Precision-Recall AUC (исправленная версия)
        batch_precision, batch_recall, _ = precision_recall_curve(y_test_numeric, batch_probs)
        online_precision, online_recall, _ = precision_recall_curve(y_test_numeric, online_probs)
        
        batch_pr_auc = auc(batch_recall, batch_precision)
        online_pr_auc = auc(online_recall, online_precision)

        print("Batch Model:")
        print(f"  AUC-ROC: {batch_auc:.4f}")
        print(f"  AUC-PR:  {batch_pr_auc:.4f}")
        
        print("Online Model:")
        print(f"  AUC-ROC: {online_auc:.4f}")
        print(f"  AUC-PR:  {online_pr_auc:.4f}")

    except Exception as e:
        print(f"Ошибка при оценке качества: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 4. Оценка задержки ---
    print("\nОценка задержки инференса...")
    n_samples = 100

    try:
        batch_sample = X_test_batch.head(1)
        online_sample = X_test_online.head(1)
        
        start_time = time.time()
        for _ in range(n_samples):
            batch_model.predict(batch_sample)
        batch_latency = (time.time() - start_time) * 1000 / n_samples

        start_time = time.time()
        for _ in range(n_samples):
            online_model.predict(online_sample)
        online_latency = (time.time() - start_time) * 1000 / n_samples

        print(f"Batch Model -> Latency: {batch_latency:.4f} ms/предсказание")
        print(f"Online Model -> Latency: {online_latency:.4f} ms/предсказание")
        
    except Exception as e:
        print(f"Ошибка при оценке задержки: {e}")
        batch_latency = online_latency = 0

    # --- 5. Логирование результатов ---
    with mlflow.start_run(run_name="Strategy Comparison"):
        # Основные метрики
        mlflow.log_metric("batch_auc_roc", batch_auc)
        mlflow.log_metric("online_auc_roc", online_auc)
        mlflow.log_metric("batch_auc_pr", batch_pr_auc)
        mlflow.log_metric("online_auc_pr", online_pr_auc)
        
        # Задержка
        mlflow.log_metric("batch_latency_ms", batch_latency)
        mlflow.log_metric("online_latency_ms", online_latency)
        
        # Параметры
        mlflow.log_param("batch_model_version", batch_model_info.version)
        mlflow.log_param("online_model_version", online_model_info.version)
        
        print("Результаты сравнения залогированы в MLflow.")


if __name__ == "__main__":
    compare_models()