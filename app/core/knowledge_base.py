# app/core/knowledge_base.py

DISEASE_INFO = {
    "OSCC": {
        "description": "Oral Squamous Cell Carcinoma (OSCC) is the most common type of malignant cancer of the oral cavity.",
        "common_causes": ["Tobacco use", "Heavy alcohol consumption", "HPV infection"],
        "risk_level": "High"
    },
    "Leukoplakia": {
        "description": "Leukoplakia is a pre-cancerous condition characterized by white patches in the mouth, which can have potential for malignant transformation.",
        "common_causes": ["Chronic irritation from tobacco", "Heavy alcohol use", "Rough teeth or dentures"],
        "risk_level": "Moderate (Pre-Malignant)"
    },
    "Normal": {
        "description": "The tissue appears to be within the range of normal oral mucosa.",
        "common_causes": [],
        "risk_level": "Low"
    }
}

TREATMENT_GUIDELINES = {
    "OSCC": {
        "Stage I": "Primary treatment is typically surgical excision.",
        "Stage II": "Treatment involves surgery, often followed by radiation therapy.",
        "Stage III": "A combination of surgery, radiation therapy, and potentially chemotherapy is the standard of care.",
        "Stage IVA": "Comprehensive treatment including surgical resection, followed by chemoradiation is usually required.",
        "Stage IVB": "Treatment is complex and focuses on both local and distant disease control.",
        "Default": "Treatment for OSCC is stage-dependent. Consultation with an oncologist is mandatory."
    },
    "Leukoplakia": {
        "Default": "Management involves eliminating causative factors and potential surgical removal. Regular, long-term clinical follow-up is essential."
    },
    "Normal": {
        "Default": "No immediate treatment is indicated. Continue with regular dental check-ups and maintain good oral hygiene."
    }
}

def generate_full_report(predicted_disease, predicted_stage, predicted_prognosis, patient_data):
    report = {
        "disease_prediction": predicted_disease,
        "disease_info": DISEASE_INFO.get(predicted_disease, {}),
        "predicted_stage": "N/A",
        "predicted_prognosis": "N/A",
        "treatment_guideline": "N/A",
        "personalized_recommendations": []
    }

    if predicted_disease == "OSCC":
        report["predicted_stage"] = predicted_stage
        report["predicted_prognosis"] = predicted_prognosis
        report["treatment_guideline"] = TREATMENT_GUIDELINES["OSCC"].get(predicted_stage, TREATMENT_GUIDELINES["OSCC"]["Default"])
    else:
        if predicted_disease in TREATMENT_GUIDELINES:
            report["treatment_guideline"] = TREATMENT_GUIDELINES[predicted_disease]["Default"]
        
    if patient_data.get('tobacco_smoking_status', 'No').lower() != 'no' and patient_data.get('tobacco_smoking_status', 'No').lower() != 'lifelong non-smoker':
        report['personalized_recommendations'].append("Immediate smoking cessation is strongly advised.")
    if patient_data.get('alcohol_history') == 'Yes':
        report['personalized_recommendations'].append("Reducing or eliminating alcohol consumption is critical.")

    if predicted_disease != "Normal":
        report['personalized_recommendations'].append("A consultation with a specialist for definitive diagnosis and treatment planning is mandatory.")
    else:
        report['personalized_recommendations'].append("Maintain good oral hygiene and continue with regular dental check-ups.")

    return report