# app/models/analysis_loader.py
import tensorflow as tf
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def create_stage3a_resources(df_full):
    # Create the specific DataFrame needed for the staging model
    target = 'ajcc_pathologic_stage'
    features = ['gender', 'race', 'age_at_index', 'tobacco_smoking_status', 'alcohol_history', 'primary_diagnosis', 'tumor_grade', 'ajcc_pathologic_t', 'ajcc_pathologic_n', 'ajcc_pathologic_m']
    df_stage = df_full[features + [target]].copy()
    
    # Clean the data
    df_stage.dropna(subset=[target], inplace=True)
    stage_counts = df_stage[target].value_counts()
    stages_to_keep = stage_counts[stage_counts > 5].index
    df_stage = df_stage[df_stage[target].isin(stages_to_keep)]
    
    X = df_stage[features]
    y = df_stage[target]
    
    label_encoder = LabelEncoder().fit(y)
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]).fit(X)
    
    return preprocessor, label_encoder

def create_stage3b_resources(df_full):
    # Create the specific DataFrame needed for the prognosis model
    target = 'vital_status'
    features = ['gender', 'race', 'age_at_index', 'tobacco_smoking_status', 'alcohol_history', 'primary_diagnosis', 'tumor_grade', 'ajcc_pathologic_stage']
    df_prognosis = df_full[features + [target]].copy()

    # Clean the data
    df_prognosis.dropna(subset=[target, 'ajcc_pathologic_stage'], inplace=True)
    df_prognosis = df_prognosis[df_prognosis[target].isin(['Alive', 'Dead'])]
    
    X = df_prognosis[features]
    y = df_prognosis[target]

    label_encoder = LabelEncoder().fit(y)
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['number']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]).fit(X)
    
    return preprocessor, label_encoder

# --- Main Loading Logic ---
try:
    # --- STAGE 3 LOADERS (Rebuilt from single source) ---
    print("--- Rebuilding Stage 3 resources from source CSV ---")
    source_df = pd.read_csv('models_store/stage3/cleaned_tcga_clinical_data.csv')
    
    preprocessor_stage3a, label_encoder_stage3a = create_stage3a_resources(source_df)
    preprocessor_stage3b, label_encoder_stage3b = create_stage3b_resources(source_df)
    
    model_stage3a = xgb.XGBClassifier()
    model_stage3a.load_model('models_store/stage3/stage_prediction_model.json')
    
    model_stage3b = xgb.XGBClassifier()
    model_stage3b.load_model('models_store/stage3/prognosis_prediction_model.json')
    
    print("--- Analysis: Stage 3 Models & Resources Loaded/Rebuilt Successfully ---")
except Exception as e:
    model_stage3a, model_stage3b, preprocessor_stage3a, label_encoder_stage3a, preprocessor_stage3b, label_encoder_stage3b = None, None, None, None, None, None
    print(f"--- Analysis ERROR: Could not load/rebuild Stage 3 resources: {e} ---")

try:
    # --- STAGE 2 LOADER ---
    model_stage2 = tf.keras.models.load_model('models_store/stage2/stage2_subtype_classifier_unbiased.keras')
    print("--- Analysis: Stage 2 Model Loaded ---")
except Exception as e:
    model_stage2 = None
    print(f"--- Analysis ERROR: Could not load Stage 2 model: {e} ---")
