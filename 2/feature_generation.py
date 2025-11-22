import pandas as pd
from datetime import datetime

df = pd.read_csv("../data/hospital_readmissions_30k.csv")
print(f"Загружено данных: {len(df)} записей")

df['systolic_bp'] = df['blood_pressure'].str.split('/').str[0].astype(int)
df['diastolic_bp'] = df['blood_pressure'].str.split('/').str[1].astype(int)

# 1. Базовые медицинские фичи
df['bp_ratio'] = df['systolic_bp'] / df['diastolic_bp']
df['bmi_category'] = pd.cut(df['bmi'], 
                           bins=[0, 18.5, 25, 30, 100], 
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
df['age_group'] = pd.cut(df['age'], 
                        bins=[0, 30, 45, 60, 75, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])

# 2. Комбинированные медицинские фичи
df['high_bp_high_chol'] = ((df['systolic_bp'] > 140) & (df['cholesterol'] > 200)).astype(int)
df['multiple_conditions'] = ((df['diabetes'] == 'Yes').astype(int) + 
                            (df['hypertension'] == 'Yes').astype(int))

# 3. Взаимодействия признаков
df['age_medication_interaction'] = df['age'] * df['medication_count']
df['bmi_cholesterol_interaction'] = df['bmi'] * df['cholesterol']

# 4. Медицинские индексы
df['cardiovascular_risk'] = (df['systolic_bp'] / 100) + (df['cholesterol'] / 200) + (df['bmi'] / 10)
df['treatment_intensity'] = df['medication_count'] * df['length_of_stay']

# 5. Бинарные фичи
df['high_cholesterol'] = (df['cholesterol'] > 200).astype(int)
df['high_bmi'] = (df['bmi'] > 30).astype(int)
df['extended_stay'] = (df['length_of_stay'] > 5).astype(int)

print(f"Создано фич: {df.shape[1]}")

# Финальный датасет для Feast
feast_data = df[[
    'patient_id', 'age', 'cholesterol', 'bmi', 'medication_count', 
    'length_of_stay', 'systolic_bp', 'diastolic_bp', 'gender', 
    'diabetes', 'hypertension', 'discharge_destination', 'readmitted_30_days',
    'bp_ratio', 'bmi_category', 'age_group', 'high_bp_high_chol', 
    'multiple_conditions', 'age_medication_interaction', 
    'bmi_cholesterol_interaction', 'cardiovascular_risk', 
    'treatment_intensity', 'high_cholesterol', 'high_bmi', 'extended_stay'
]].copy()

# Добавляем timestamp (используем текущее время, так как в данных нет временных меток)
current_time = datetime.now()
feast_data['event_timestamp'] = current_time
feast_data['created_timestamp'] = current_time

print("Финальные фичи:")
for i, col in enumerate(feast_data.columns, 1):
    print(f"  {i:2d}. {col}")

feast_data.to_parquet("./hospital_readmissions/feature_repo/data/patient_features.parquet", index=False)

print(f"\nФайл с фичами для Feast создан: {len(feast_data)} записей")
print(f"Размерность данных: {feast_data.shape}")