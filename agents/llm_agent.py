import os
import requests

API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"

HF_TOKEN = os.getenv("HF_TOKEN")


def generate_llm_explanation(
    prediction,
    confidence,
    metadata_found,
    watermark_found,
    extracted_text
):

    prompt = f"""
You are a digital forensic expert.

Prediction: {prediction}
Confidence: {confidence}%
Metadata Found: {metadata_found}
Watermark Found: {watermark_found}
OCR Text: {extracted_text}

Write a professional explanation in about 120 words.
"""

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": prompt
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return "Unable to generate AI explanation."

    result = response.json()

    if isinstance(result, list):
        return result[0]["generated_text"]

    return "Unable to generate AI explanation."