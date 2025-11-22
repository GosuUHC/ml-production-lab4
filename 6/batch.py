import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import mlflow
from mlflow.tracking import MlflowClient

# --- Настройки ---
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Hospital Readmission Strategy Comparison")

FEAST_REPO_PATH = "../2/hospital_readmissions/feature_repo"
MODEL_NAME = "HospitalReadmissionModel-Batch"


def train_batch_model():
    print("--- Запуск пакетного обучения (Batch Training) ---")

    client = MlflowClient()

    # --- 1. Получение данных для обучения ---
    print("Загрузка данных из parquet файла...")

    try:
        # Загружаем данные напрямую из parquet файла
        df = pd.read_parquet(
            f'{FEAST_REPO_PATH}/data/patient_features.parquet')

        if df.empty:
            print("Ошибка: Parquet файл пустой")
            return

        print(f"Загружено {len(df)} записей из parquet файла")

    except Exception as e:
        print(f"Ошибка при загрузке данных из parquet файла: {e}")
        return

    # Обрабатываем данные
    training_df = df.dropna()

    if training_df.empty:
        print("Предупреждение: После очистки NaN не осталось данных для обучения.")
        return

    print(f"Осталось {len(training_df)} записей после очистки")

    # Преобразуем категориальные фичи в one-hot encoding
    categorical_columns = ['gender', 'diabetes', 'hypertension',
                           'discharge_destination', 'bmi_category', 'age_group']

    # Преобразуем бинарные категориальные фичи
    binary_mapping = {'Yes': 1, 'No': 0}
    for col in ['diabetes', 'hypertension']:
        if col in training_df.columns:
            training_df[col] = training_df[col].map(binary_mapping).fillna(0)

    # One-hot encoding для остальных категориальных фич
    training_df_encoded = pd.get_dummies(
        training_df, columns=categorical_columns, prefix=categorical_columns)

    # Убираем служебные колонки
    columns_to_drop = ['patient_id', 'event_timestamp',
                       'created_timestamp', 'readmitted_30_days']
    feature_columns = [col for col in training_df_encoded.columns
                       if col not in columns_to_drop]

    X_train = training_df_encoded[feature_columns]
    y_train = training_df_encoded['readmitted_30_days']

    print(
        f"Размерность данных: X_train {X_train.shape}, y_train {y_train.shape}")
    print(
        f"Баланс классов: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"🔧 Количество фич: {len(feature_columns)}")

    # --- 2. Обучение модели ---
    print("🔧 Обучение модели GradientBoosting для бинарной классификации...")

    # Параметры для GradientBoosting Classifier
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 3,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'subsample': 0.8,
        'random_state': 42
    }

    model = GradientBoostingClassifier(**params)

    # Разделяем на train/validation для оценки качества
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    print(
        f"Обучение на {len(X_train_split)} samples, валидация на {len(X_val)} samples")

    model.fit(X_train_split, y_train_split)

    # Оценка модели
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_pred_proba)
    accuracy = accuracy_score(y_val, y_pred)

    print(
        f"Модель обучена. Validation AUC: {auc:.4f}, Accuracy: {accuracy:.4f}")

    # --- 3. Логирование в MLflow ---
    with mlflow.start_run(run_name="Batch Training Run") as run:
        # Логируем параметры
        mlflow.log_params(params)
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("feature_count", len(feature_columns))
        mlflow.log_param("data_source", "parquet_file")

        # Логируем метрики (только AUC и Accuracy)
        mlflow.log_metrics({
            "val_auc": auc,
            "val_accuracy": accuracy,
            "train_auc": roc_auc_score(y_train_split, model.predict_proba(X_train_split)[:, 1]),
            "train_accuracy": accuracy_score(y_train_split, model.predict(X_train_split))
        })

        # Логируем информацию о данных
        mlflow.log_param("class_balance", str(
            y_train.value_counts(normalize=True).to_dict()))
        mlflow.log_param("feature_names_count", len(feature_columns))

        # Логируем модель
        input_example = X_train.head(5)
        signature = mlflow.models.infer_signature(
            input_example, model.predict(input_example))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=input_example,
            signature=signature
        )

        model_uri = f"runs:/{run.info.run_id}/model"
        print(f"Модель залогирована: {model_uri}")

        # --- 4. "Деплой" в Staging (для Canary) ---
        print("Регистрация новой версии и перевод в 'Staging'...")
        try:
            # Даем время MLflow зарегистрировать модель
            import time
            time.sleep(2)

            # Ищем последнюю созданную версию
            latest_version_info = client.get_latest_versions(
                MODEL_NAME, stages=["None"])[0]
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=latest_version_info.version,
                stage="Staging",
                archive_existing_versions=True  # Архивирует предыдущую модель в Staging
            )
            print(
                f"Модель версии {latest_version_info.version} переведена в 'Staging'.")

            # Логируем информацию о деплое
            mlflow.log_param("deployed_version", latest_version_info.version)
            mlflow.log_param("deployment_stage", "Staging")

        except IndexError:
            print(
                f"Не найдено новых версий модели '{MODEL_NAME}' для перевода в Staging.")
        except Exception as e:
            print(f"Ошибка при переводе модели в Staging: {e}")

    print("Пакетное обучение завершено успешно!")


if __name__ == "__main__":
    train_batch_model()
