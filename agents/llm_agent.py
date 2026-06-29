import ollama


def generate_llm_explanation(
    prediction,
    confidence,
    risk_level,
    metadata_found,
    watermark_found,
    extracted_text
):

    prompt = f"""
You are an AI Digital Forensics Expert.

Analyze the following forensic results.

Prediction:
{prediction}

Confidence:
{confidence}%

Risk Level:
{risk_level}

Metadata Present:
{metadata_found}

Watermark Detected:
{watermark_found}

Extracted Text:
{extracted_text}

Generate a professional forensic explanation.

Mention:

• Why the model predicted this class.

• Explain confidence score.

• Explain metadata findings.

• Explain watermark findings.

• Explain OCR findings.

• Mention that Grad-CAM highlights important regions.

Keep the explanation between 150 and 200 words.

Use professional language.
"""

    response = ollama.chat(

        model="gemma3:4b",

        messages=[

            {
                "role": "user",
                "content": prompt
            }

        ]

    )

    return response["message"]["content"]