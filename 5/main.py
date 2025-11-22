# monitoring_pipeline.py

import os
import requests
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import datetime
import json
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping

# --- 1. Настройки ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Hospital Readmission Monitoring"
mlflow.set_experiment(EXPERIMENT_NAME)

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

# Пороги для бинарной классификации
MODEL_PERFORMANCE_DEGRADATION_AUC_THRESHOLD = 0.95  # 5% деградация AUC
RETRAIN_PERFORMANCE_DEGRADATION_AUC_THRESHOLD = 0.90  # 10% деградация AUC
DATA_DRIFT_THRESHOLD = 0.5  # 50% смещенных фич

# --- 2. Функции для алертинга ---


def send_alert(message: str, is_critical: bool = False):
    prefix = "🚨 *Критический алерт* 🚨" if is_critical else "⚠️ *Предупреждение* ⚠️"
    full_message = f"{prefix}\n{message}"

    print(full_message)

    if not SLACK_WEBHOOK_URL:
        print("Переменная SLACK_WEBHOOK_URL не установлена. Алерт не отправлен в Slack.")
        return

    try:
        payload = {"blocks": [{"type": "section", "text": {
            "type": "mrkdwn", "text": full_message}}]}
        requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        print("Алерт успешно отправлен в Slack.")
    except Exception as e:
        print(f"Ошибка при отправке алерта в Slack: {e}")

# --- 3. Функции мониторинга для бинарной классификации ---


def monitor_data_drift(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """Генерирует отчет о дрифте данных для медицинских фич."""
    print("\n--- Запуск мониторинга дрифта данных ---")

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(reference_data=reference_data,
                          current_data=current_data)

    report_dict = json.loads(snapshot.json())

    try:
        drift_metric_value = report_dict['metrics'][0]['value']
        print(drift_metric_value)
        num_drifted_columns = int(drift_metric_value['count'])
        dataset_drift_detected = num_drifted_columns > 0

        # Вычисляем процент смещенных фич
        total_columns = len(reference_data.columns)
        drift_percentage = num_drifted_columns / \
            total_columns if total_columns > 0 else 0

    except (KeyError, IndexError, TypeError) as e:
        print(f"Ошибка при извлечении результатов дрифта из отчета: {e}")
        return {"dataset_drift": False, "drifted_columns": 0, "drift_percentage": 0}

    with mlflow.start_run(run_name="Data Drift Report"):
        snapshot.save_html("data_drift_report.html")
        mlflow.log_artifact("data_drift_report.html", "reports")
        mlflow.log_dict(report_dict, "data_drift_report.json")

        mlflow.log_metric("num_drifted_columns", num_drifted_columns)
        mlflow.log_metric("drift_percentage", drift_percentage)
        mlflow.log_metric("dataset_drift", int(dataset_drift_detected))

    print(
        f"Обнаружен дрифт в {num_drifted_columns} колонках ({drift_percentage:.1%})")

    if drift_percentage > DATA_DRIFT_THRESHOLD:
        send_alert(
            f"Критический дрифт данных! {drift_percentage:.1%} фич смещено ({num_drifted_columns}/{total_columns}).",
            is_critical=True
        )
    elif dataset_drift_detected:
        send_alert(
            f"Обнаружен дрифт данных! Количество смещенных колонок: {num_drifted_columns}.")

    return {
        "dataset_drift": dataset_drift_detected,
        "drifted_columns": num_drifted_columns,
        "drift_percentage": drift_percentage
    }


def monitor_model_performance(model, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> dict:
    """Генерирует отчет о производительности модели для бинарной классификации."""
    print("\n--- Запуск мониторинга производительности модели ---")

    # Создаем копии данных
    ref_data_copy = reference_data.copy()
    curr_data_copy = current_data.copy()

    # Получаем предсказания
    ref_predictions = model.predict(
        ref_data_copy.drop('readmitted_30_days', axis=1))
    curr_predictions = model.predict(
        curr_data_copy.drop('readmitted_30_days', axis=1))

    # Подготавливаем данные для evidently
    ref_data_copy['target'] = ref_data_copy['readmitted_30_days']
    curr_data_copy['target'] = curr_data_copy['readmitted_30_days']
    ref_data_copy.drop('readmitted_30_days', axis=1, inplace=True)
    curr_data_copy.drop('readmitted_30_days', axis=1, inplace=True)

    ref_data_copy['prediction'] = ref_predictions
    curr_data_copy['prediction'] = curr_predictions

    # Настраиваем column mapping для классификации
    from evidently.legacy.pipeline.column_mapping import TaskType

    column_mapping = ColumnMapping(
        target='target',
        prediction='prediction',
        task=TaskType.CLASSIFICATION_TASK
    )

    report = Report(metrics=[ClassificationPreset()])

    try:
        snapshot = report.run(
            reference_data=ref_data_copy,
            current_data=curr_data_copy,
            column_mapping=column_mapping
        )
    except Exception as e:
        print(f"Ошибка при генерации отчета о качестве: {e}")
        return {"reference_auc": -1, "current_auc": -1, "reference_f1": -1, "current_f1": -1}

    report_dict = json.loads(snapshot.json())

    try:
        # Извлекаем метрики классификации
        quality_metrics = {}
        for metric in report_dict['metrics']:
            if metric['metric'] == 'ClassificationQualityMetric':
                quality_metrics = metric['result']
                break

        ref_auc = quality_metrics.get('reference', {}).get('roc_auc', -1)
        curr_auc = quality_metrics.get('current', {}).get('roc_auc', -1)
        ref_f1 = quality_metrics.get('reference', {}).get('f1', -1)
        curr_f1 = quality_metrics.get('current', {}).get('f1', -1)

    except (KeyError, IndexError) as e:
        print(f"Ошибка при извлечении метрик качества из отчета: {e}")
        print("--- СТРУКТУРА JSON ОТЧЕТА О КАЧЕСТВЕ ---")
        print(json.dumps(report_dict, indent=4))
        return {"reference_auc": -1, "current_auc": -1, "reference_f1": -1, "current_f1": -1}

    with mlflow.start_run(run_name="Model Performance Report"):
        snapshot.save_html("model_performance_report.html")
        mlflow.log_artifact("model_performance_report.html", "reports")
        mlflow.log_dict(report_dict, "model_performance_report.json")
        mlflow.log_metrics({
            "reference_auc": ref_auc,
            "current_auc": curr_auc,
            "reference_f1": ref_f1,
            "current_f1": curr_f1
        })

    print(f"Reference AUC: {ref_auc:.3f}, Current AUC: {curr_auc:.3f}")
    print(f"Reference F1: {ref_f1:.3f}, Current F1: {curr_f1:.3f}")

    # Проверяем деградацию производительности
    if ref_auc > 0 and curr_auc > 0:
        auc_ratio = curr_auc / ref_auc

        if auc_ratio < MODEL_PERFORMANCE_DEGRADATION_AUC_THRESHOLD:
            degradation = (1 - auc_ratio) * 100
            send_alert(
                f"Обнаружена деградация модели! AUC снизился на {degradation:.1f}% (с {ref_auc:.3f} до {curr_auc:.3f}).")

    return {
        "reference_auc": ref_auc,
        "current_auc": curr_auc,
        "reference_f1": ref_f1,
        "current_f1": curr_f1
    }

# --- 4. Логика ретрейна ---


def retrain_model():
    """Запуск процедуры переобучения модели."""
    send_alert("Запущена процедура автоматического переобучения модели предсказания повторной госпитализации.", is_critical=True)

    # Здесь будет логика переобучения
    # Например, запуск тренировочного скрипта
    print("Запуск переобучения модели...")


def check_and_run_retrain(data_drift_info: dict, model_performance_info: dict):
    """Проверяет необходимость переобучения и запускает его при необходимости."""
    print("\n--- Проверка необходимости переобучения ---")
    retrain_needed = False
    reason = ""

    # Плановое еженедельное переобучение
    if datetime.date.today().weekday() == 0:  # Понедельник
        retrain_needed = True
        reason = "Плановое еженедельное переобучение."

    # Критический дрифт данных
    drift_percentage = data_drift_info.get('drift_percentage', 0)
    if drift_percentage > DATA_DRIFT_THRESHOLD and not retrain_needed:
        retrain_needed = True
        reason = f"Критический дрифт данных ({drift_percentage:.1%} фич смещено)."

    # Критическая деградация производительности
    ref_auc = model_performance_info.get('reference_auc', -1)
    curr_auc = model_performance_info.get('current_auc', -1)
    if (ref_auc > 0 and curr_auc > 0 and
        curr_auc < ref_auc * RETRAIN_PERFORMANCE_DEGRADATION_AUC_THRESHOLD and
            not retrain_needed):
        degradation = (1 - (curr_auc / ref_auc)) * 100
        retrain_needed = True
        reason = f"Критическая деградация производительности (AUC снизился на {degradation:.1f}%)."

    if retrain_needed:
        print(f"Принято решение о переобучении. Причина: {reason}")
        retrain_model()
    else:
        print("Переобучение не требуется.")


# --- 5. Основной пайплайн ---
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_data_path = os.path.join(
        base_dir, '..', '3', 'monitoring', 'data', 'reference_data.parquet')
    curr_data_path = os.path.join(
        base_dir, '..', '3', 'monitoring', 'data', 'current_data.parquet')

    try:
        ref_data = pd.read_parquet(ref_data_path)
        curr_data = pd.read_parquet(curr_data_path)
        print(
            f"Загружены данные: reference={len(ref_data)} записей, current={len(curr_data)} записей")
    except FileNotFoundError:
        print(
            f"Ошибка: Файлы для мониторинга не найдены по путям:\n{ref_data_path}\n{curr_data_path}")
        print("Создайте тестовые данные с помощью create_monitoring_data.py")
        exit()

    # Загружаем production модель
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(
            "HospitalReadmissionModel", stages=["Production"])
        if not latest_versions:
            raise IndexError("No model versions found in Production stage.")
        prod_model_info = latest_versions[0]
        model_uri = f"models:/{prod_model_info.name}/{prod_model_info.version}"
        production_model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Загружена production модель: v{prod_model_info.version}")
    except IndexError as e:
        print(
            f"❌ Ошибка: {e}. Убедитесь, что хотя бы одна версия модели имеет стейдж 'Production'.")
        exit()
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        exit()

    # Запускаем мониторинг
    data_drift_results = monitor_data_drift(
        ref_data.drop('readmitted_30_days', axis=1),
        curr_data.drop('readmitted_30_days', axis=1)
    )

    model_performance_results = monitor_model_performance(
        production_model,
        ref_data,
        curr_data
    )

    # Проверяем необходимость переобучения
    check_and_run_retrain(data_drift_results, model_performance_results)

    print("\n✅ Мониторинг завершен!")
