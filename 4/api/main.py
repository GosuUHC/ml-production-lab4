# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from feast import FeatureStore
import os
from datetime import datetime
import random
import logging

# --- Настройка логирования для Shadow Mode ---
logging.basicConfig(
    filename='shadow_mode_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

# --- Настройка ---
MLFLOW_TRACKING_URI = 'http://localhost:5000'
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
FEAST_REPO_PATH = '../../2/hospital_readmissions/feature_repo'

# --- Параметры безопасного деплоя ---
CANARY_TRAFFIC_PERCENT = 5  # 5% трафика на новую модель

app = FastAPI(title="Hospital Readmission Prediction API")

# --- Глобальные объекты ---
try:
    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    # Смотрим, что зарегистрировано
    print("entities:")
    for entity in store.list_entities():
        print(f"  - {entity.name}")

    print("\nfeature views:")
    for fv in store.list_feature_views():
        print(f"  - {fv.name}")
        for feature in fv.features:
            print(f"    * {feature.name} ({feature.dtype})")

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    # 1. Загружаем Production модель
    prod_model_info = client.get_latest_versions(
        "HospitalReadmissionModel", stages=["Production"])[0]
    prod_model_uri = f"models:/HospitalReadmissionModel/{prod_model_info.version}"
    prod_model = mlflow.pyfunc.load_model(prod_model_uri)
    PRODUCTION_MODEL_VERSION = prod_model_info.version

    if hasattr(prod_model, '_model_impl'):
        print("Доступ к внутренней реализации модели...")
        # Для sklearn моделей
        if hasattr(prod_model._model_impl, 'sklearn_model'):
            sklearn_model = prod_model._model_impl.sklearn_model
            if hasattr(sklearn_model, 'feature_names_in_'):
                print("🎯 Фичи, которые ожидает модель:")
                print(sklearn_model.feature_names_in_)

    # 2. Загружаем Staging (Canary) модель
    staging_model = None
    STAGING_MODEL_VERSION = None
    try:
        staging_model_info = client.get_latest_versions(
            "HospitalReadmissionModel", stages=["Staging"])[0]
        staging_model_uri = f"models:/HospitalReadmissionModel/{staging_model_info.version}"
        staging_model = mlflow.pyfunc.load_model(staging_model_uri)
        STAGING_MODEL_VERSION = staging_model_info.version
        print(
            f"Successfully loaded Production model v{PRODUCTION_MODEL_VERSION} and Staging model v{STAGING_MODEL_VERSION}")
    except IndexError:
        print(
            f"Successfully loaded Production model v{PRODUCTION_MODEL_VERSION}. No model found in Staging.")

except Exception as e:
    raise RuntimeError(f"Failed to initialize models or feature store: {e}")


class ReadmissionRequest(BaseModel):
    patient_id: int


class ReadmissionResponse(BaseModel):
    patient_id: int
    prediction: int
    model_version: str


EXPECTED_MODEL_FEATURES = [
    'age', 'cholesterol', 'bmi', 'medication_count', 'length_of_stay',
    'systolic_bp', 'diastolic_bp', 'diabetes', 'hypertension', 'bp_ratio',
    'high_bp_high_chol', 'multiple_conditions', 'age_medication_interaction',
    'bmi_cholesterol_interaction', 'cardiovascular_risk', 'treatment_intensity',
    'high_cholesterol', 'high_bmi', 'extended_stay', 'gender_Female',
    'gender_Male', 'gender_Other', 'discharge_destination_Home',
    'discharge_destination_Nursing_Facility', 'discharge_destination_Rehab',
    'bmi_category_Underweight', 'bmi_category_Normal', 'bmi_category_Overweight',
    'bmi_category_Obese', 'age_group_Young', 'age_group_Adult', 'age_group_Middle',
    'age_group_Senior', 'age_group_Elderly'
]


def preprocess_features_for_model(raw_features: pd.DataFrame) -> pd.DataFrame:
    """Преобразуем сырые фичи из Feature Store в формат модели"""

    processed_features = {}

    # 1. Числовые фичи из patient_stats (копируем как есть)
    patient_stats_features = [
        'age', 'cholesterol', 'bmi', 'medication_count', 'length_of_stay',
        'systolic_bp', 'diastolic_bp', 'bp_ratio', 'cardiovascular_risk',
        'treatment_intensity', 'high_cholesterol', 'high_bmi', 'extended_stay'
    ]

    for feature in patient_stats_features:
        if feature in raw_features.columns:
            processed_features[feature] = raw_features[feature].iloc[0]
        else:
            # Значения по умолчанию для числовых фич
            defaults = {
                'age': 50, 'cholesterol': 200, 'bmi': 25.0, 'medication_count': 3,
                'length_of_stay': 5, 'systolic_bp': 120, 'diastolic_bp': 80,
                'bp_ratio': 1.5, 'cardiovascular_risk': 3.0, 'treatment_intensity': 15,
                'high_cholesterol': 0, 'high_bmi': 0, 'extended_stay': 0
            }
            processed_features[feature] = defaults.get(feature, 0)

    # 2. Фичи из interaction_features
    interaction_features = [
        'high_bp_high_chol', 'multiple_conditions', 'age_medication_interaction',
        'bmi_cholesterol_interaction'
    ]

    for feature in interaction_features:
        if feature in raw_features.columns:
            processed_features[feature] = raw_features[feature].iloc[0]
        else:
            # Вычисляем или устанавливаем по умолчанию
            if feature == 'high_bp_high_chol':
                systolic = processed_features.get('systolic_bp', 120)
                cholesterol = processed_features.get('cholesterol', 200)
                processed_features[feature] = 1 if (
                    systolic > 140) and (cholesterol > 200) else 0
            elif feature == 'multiple_conditions':
                # Предполагаем, что есть diabetes и hypertension
                processed_features[feature] = 1  # по умолчанию
            elif feature == 'age_medication_interaction':
                age = processed_features.get('age', 50)
                med_count = processed_features.get('medication_count', 3)
                processed_features[feature] = age * med_count
            elif feature == 'bmi_cholesterol_interaction':
                bmi = processed_features.get('bmi', 25.0)
                cholesterol = processed_features.get('cholesterol', 200)
                processed_features[feature] = bmi * cholesterol
            else:
                processed_features[feature] = 0

    # 3. Категориальные фичи из demographic_features - преобразуем в one-hot encoding
    # gender
    if 'gender' in raw_features.columns:
        gender = raw_features['gender'].iloc[0]
        processed_features['gender_Female'] = 1 if gender == 'Female' else 0
        processed_features['gender_Male'] = 1 if gender == 'Male' else 0
        processed_features['gender_Other'] = 1 if gender == 'Other' else 0
    else:
        processed_features['gender_Female'] = 0
        processed_features['gender_Male'] = 1  # по умолчанию Male
        processed_features['gender_Other'] = 0

    # diabetes и hypertension
    for feature in ['diabetes', 'hypertension']:
        if feature in raw_features.columns:
            value = raw_features[feature].iloc[0]
            processed_features[feature] = 0 if value == 'No' else 1
        else:
            processed_features[feature] = 0  # по умолчанию

    # discharge_destination
    if 'discharge_destination' in raw_features.columns:
        destination = raw_features['discharge_destination'].iloc[0]
        processed_features['discharge_destination_Home'] = 1 if destination == 'Home' else 0
        processed_features['discharge_destination_Nursing_Facility'] = 1 if destination == 'Nursing_Facility' else 0
        # по умолчанию нет Rehab
        processed_features['discharge_destination_Rehab'] = 0
    else:
        processed_features['discharge_destination_Home'] = 1
        processed_features['discharge_destination_Nursing_Facility'] = 0
        processed_features['discharge_destination_Rehab'] = 0

    # bmi_category
    if 'bmi_category' in raw_features.columns:
        bmi_cat = raw_features['bmi_category'].iloc[0]
        processed_features['bmi_category_Underweight'] = 1 if bmi_cat == 'Underweight' else 0
        processed_features['bmi_category_Normal'] = 1 if bmi_cat == 'Normal' else 0
        processed_features['bmi_category_Overweight'] = 1 if bmi_cat == 'Overweight' else 0
        processed_features['bmi_category_Obese'] = 1 if bmi_cat == 'Obese' else 0
    else:
        processed_features['bmi_category_Normal'] = 1  # по умолчанию Normal
        processed_features['bmi_category_Underweight'] = 0
        processed_features['bmi_category_Overweight'] = 0
        processed_features['bmi_category_Obese'] = 0

    # age_group
    if 'age_group' in raw_features.columns:
        age_grp = raw_features['age_group'].iloc[0]
        processed_features['age_group_Young'] = 1 if age_grp == 'Young' else 0
        processed_features['age_group_Adult'] = 1 if age_grp == 'Adult' else 0
        processed_features['age_group_Middle'] = 1 if age_grp == 'Middle' else 0
        processed_features['age_group_Senior'] = 1 if age_grp == 'Senior' else 0
        processed_features['age_group_Elderly'] = 1 if age_grp == 'Elderly' else 0
    else:
        processed_features['age_group_Adult'] = 1  # по умолчанию Adult
        processed_features['age_group_Young'] = 0
        processed_features['age_group_Middle'] = 0
        processed_features['age_group_Senior'] = 0
        processed_features['age_group_Elderly'] = 0

    # Создаем DataFrame с правильным порядком колонок
    final_df = pd.DataFrame([processed_features])[EXPECTED_MODEL_FEATURES]

    print(f"✅ После preprocessing: {len(final_df.columns)} фич")
    return final_df


def get_features_from_store(patient_id: int) -> pd.DataFrame:
    """Получение фичей из Feature Store для пациента"""
    try:
        # Создаем entity dataframe
        entity_df = pd.DataFrame({
            "patient_id": [patient_id],
            "event_timestamp": [datetime.now()]  # ← ЭТО ОБЯЗАТЕЛЬНО!
        })

        # Получаем фичи из Feature Store
        features_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                # patient_stats features
                "patient_stats:age",
                "patient_stats:cholesterol",
                "patient_stats:bmi",
                "patient_stats:medication_count",
                "patient_stats:length_of_stay",
                "patient_stats:systolic_bp",
                "patient_stats:diastolic_bp",
                "patient_stats:bp_ratio",
                "patient_stats:cardiovascular_risk",
                "patient_stats:treatment_intensity",
                "patient_stats:high_cholesterol",
                "patient_stats:high_bmi",
                "patient_stats:extended_stay",

                # interaction_features
                "interaction_features:high_bp_high_chol",
                "interaction_features:multiple_conditions",
                "interaction_features:age_medication_interaction",
                "interaction_features:bmi_cholesterol_interaction",

                # demographic_features
                "demographic_features:gender",
                "demographic_features:diabetes",
                "demographic_features:hypertension",
                "demographic_features:discharge_destination",
                "demographic_features:bmi_category",
                "demographic_features:age_group"
            ]
        ).to_df()

        if features_df.empty:
            raise ValueError(f"No features found for patient {patient_id}")

        feature_columns = [col for col in features_df.columns if col not in [
            'patient_id', 'event_timestamp']]
        features_df = features_df[feature_columns]

        # Преобразуем в формат модели
        processed_features = preprocess_features_for_model(features_df)

        print(f"Успешно обработано {len(processed_features.columns)} фич")
        return processed_features

    except Exception as e:
        raise ValueError(f"Error getting features from store: {e}")


@app.post("/predict", response_model=ReadmissionResponse)
def predict(request: ReadmissionRequest):
    try:

        # --- 1. Получение фичей (один раз для обеих моделей) ---
        features_df = get_features_from_store(request.patient_id)
        
        # Убираем служебные колонки
        feature_columns = [col for col in features_df.columns if col not in [
            'patient_id', 'event_timestamp']]
        features_df = features_df[feature_columns]

        # --- 2. Логика Canary Release и Shadow Mode ---

        # Определяем, какую модель использовать для ответа пользователю
        use_canary = (staging_model is not None) and (
            random.random() < CANARY_TRAFFIC_PERCENT / 100)

        if use_canary:
            # --- CANARY PATH (5% трафика) ---
            # Новая модель обрабатывает запрос и ее результат возвращается пользователю
            prediction = staging_model.predict(features_df)[0]
            model_version_for_response = STAGING_MODEL_VERSION
        else:
            # --- PRODUCTION PATH (95% трафика) ---
            # Старая модель обрабатывает запрос
            prediction = prod_model.predict(features_df)[0]
            model_version_for_response = PRODUCTION_MODEL_VERSION

        # --- SHADOW MODE LOGIC ---
        # Если есть Staging модель, втихую прогоняем данные и через нее, чтобы сравнить результаты
        if staging_model is not None:
            # Получаем предсказания от обеих моделей (одно уже есть)
            prod_prediction = prod_model.predict(
                features_df)[0] if use_canary else prediction
            staging_prediction = prediction if use_canary else staging_model.predict(features_df)[
                0]

            # Логируем разницу для последующего анализа
            prediction_diff = prod_prediction - staging_prediction

            log_message = (
                f"PatientID: {request.patient_id}, "
                f"Prod_v{PRODUCTION_MODEL_VERSION}: pred={prod_prediction}, "
                f"Staging_v{STAGING_MODEL_VERSION}: pred={staging_prediction}, "
                f"Pred_Diff: {prediction_diff}"
            )
            logging.info(log_message)

        return {
            "patient_id": request.patient_id,
            "prediction": 0 if prediction == 'No' else 1,
            "model_version": model_version_for_response
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {e}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "production_model_loaded": prod_model is not None,
        "staging_model_loaded": staging_model is not None,
        "production_version": PRODUCTION_MODEL_VERSION,
        "staging_version": STAGING_MODEL_VERSION,
        "canary_traffic_percent": CANARY_TRAFFIC_PERCENT
    }
