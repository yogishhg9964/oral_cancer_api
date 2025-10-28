# app/api/analysis.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models import analysis_loader as loader # Import our new loader
from app.core.knowledge_base import generate_full_report
import pandas as pd
import numpy as np
import tensorflow as tf
import io

router = APIRouter()

@router.post("/analyze")
async def analyze_prediction(
    age: int = Form(...),
    gender: str = Form(...),
    race: str = Form(...),
    tobacco_smoking_status: str = Form(...),
    alcohol_history: str = Form(...),
    ajcc_pathologic_t: str = Form(...),
    ajcc_pathologic_n: str = Form(...),
    ajcc_pathologic_m: str = Form(...),
    image: UploadFile = File(...)
):
    if not all([loader.model_stage2, loader.model_stage3a, loader.model_stage3b]):
        raise HTTPException(status_code=503, detail="One or more analysis models are not loaded.")

    # --- Run Stage 2: Sub-Type Classification ---
    contents = await image.read()
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(contents), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    stage2_pred = loader.model_stage2.predict(img_array)
    class_names_stage2 = ['Leukoplakia', 'Normal', 'OSCC']
    predicted_disease = class_names_stage2[np.argmax(stage2_pred)]

    # --- Get patient data for reports ---
    patient_report_data = {
        "smoking_status": tobacco_smoking_status, 
        "alcohol_consumption": "High" if alcohol_history == "Yes" else "None"
    }

    # If not cancer, generate a simple report and return early
    if predicted_disease != 'OSCC':
        return generate_full_report(predicted_disease, None, None, patient_report_data)
    
    # --- Run Stage 3: Staging & Prognosis (only if OSCC) ---
    patient_data_for_stage3 = {
        'age_at_index': age, 'gender': gender, 'race': race,
        'tobacco_smoking_status': tobacco_smoking_status, 'alcohol_history': alcohol_history,
        'ajcc_pathologic_t': ajcc_pathologic_t, 'ajcc_pathologic_n': ajcc_pathologic_n,
        'ajcc_pathologic_m': ajcc_pathologic_m,
        'primary_diagnosis': 'Squamous cell carcinoma, NOS', 'tumor_grade': 'G2'
    }
    patient_df_stage3 = pd.DataFrame([patient_data_for_stage3])
    
    # Run Stage 3A: Staging Model
    processed_stage3a = loader.preprocessor_stage3a.transform(patient_df_stage3)
    stage_encoded = loader.model_stage3a.predict(processed_stage3a)
    predicted_stage = loader.label_encoder_stage3a.inverse_transform(stage_encoded)[0]

    # Run Stage 3B: Prognosis Model
    patient_df_stage3['ajcc_pathologic_stage'] = predicted_stage
    processed_stage3b = loader.preprocessor_stage3b.transform(patient_df_stage3)
    prognosis_encoded = loader.model_stage3b.predict(processed_stage3b)
    predicted_prognosis = loader.label_encoder_stage3b.inverse_transform(prognosis_encoded)[0]

    # --- Generate the Final Full Report ---
    return generate_full_report(predicted_disease, predicted_stage, predicted_prognosis, patient_report_data)