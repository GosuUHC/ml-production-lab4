from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.types import Float32, Int32, String, Bool
from feast.infra.offline_stores.file_source import FileSource

# Entity definition
patient = Entity(
    name="patient",
    description="A hospital patient",
    join_keys=["patient_id"]
)

# Data source
patient_features_source = FileSource(
    name="patient_features",
    path="data/patient_features.parquet",
    timestamp_field="event_timestamp"
)

# Feature Views
patient_stats_fv = FeatureView(
    name="patient_stats",
    entities=[patient],
    ttl=timedelta(days=365),
    schema=[
        Field(name="age", dtype=Int32),
        Field(name="cholesterol", dtype=Int32),
        Field(name="bmi", dtype=Float32),
        Field(name="medication_count", dtype=Int32),
        Field(name="length_of_stay", dtype=Int32),
        Field(name="systolic_bp", dtype=Int32),
        Field(name="diastolic_bp", dtype=Int32),
        Field(name="bp_ratio", dtype=Float32),
        Field(name="cardiovascular_risk", dtype=Float32),
        Field(name="treatment_intensity", dtype=Float32),
        Field(name="high_cholesterol", dtype=Int32),
        Field(name="high_bmi", dtype=Int32),
        Field(name="extended_stay", dtype=Int32)
    ],
    source=patient_features_source,
    online=True
)

demographic_fv = FeatureView(
    name="demographic_features", 
    entities=[patient],
    ttl=timedelta(days=365),
    schema=[
        Field(name="gender", dtype=String),
        Field(name="diabetes", dtype=String),
        Field(name="hypertension", dtype=String), 
        Field(name="discharge_destination", dtype=String),
        Field(name="bmi_category", dtype=String),
        Field(name="age_group", dtype=String)
    ],
    source=patient_features_source,
    online=True
)

interaction_fv = FeatureView(
    name="interaction_features",
    entities=[patient],
    ttl=timedelta(days=365),
    schema=[
        Field(name="high_bp_high_chol", dtype=Int32),
        Field(name="multiple_conditions", dtype=Int32),
        Field(name="age_medication_interaction", dtype=Float32),
        Field(name="bmi_cholesterol_interaction", dtype=Float32)
    ],
    source=patient_features_source,
    online=True
)