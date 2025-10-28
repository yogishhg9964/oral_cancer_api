from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.ml_models import model_loader
import pandas as pd
import numpy as np
import tensorflow as tf
import io

router = APIRouter()
CLASS_NAMES = ['Cancer', 'Non-Cancer']

@router.post("/predict")
async def predict(
    # We define all our form fields using FastAPI's Form
    age: int = Form(...),
    gender: str = Form(...),
    smoking_status: str = Form(...),
    alcohol_consumption: str = Form(...),
    hpv_status: str = Form(...),
    lesion_size_mm: float = Form(...),
    image: UploadFile = File(...)
):
    if model_loader.model is None or model_loader.preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    # --- 1. Process the Image Input ---
    try:
        contents = await image.read()
        img = tf.keras.preprocessing.image.load_img(io.BytesIO(contents), target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

    # --- 2. Process the Tabular Input ---
    try:
        patient_data = {
            'age': [age], 'gender': [gender], 'smoking_status': [smoking_status],
            'alcohol_consumption': [alcohol_consumption], 'hpv_status': [hpv_status],
            'lesion_size_mm': [lesion_size_mm]
        }
        patient_df = pd.DataFrame(patient_data)
        tabular_data_processed = model_loader.preprocessor.transform(patient_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing tabular data: {e}")
        
    # --- 3. Make the Prediction ---
    try:
        prediction = model_loader.model.predict({
            'image_input': img_array,
            'tabular_input': tabular_data_processed
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

    # --- 4. Format the Response ---
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(prediction[0]) * 100)
    
    return {
        "prediction": predicted_class_name,
        "confidence": round(confidence, 2)
    }