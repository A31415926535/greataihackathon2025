import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL_FINAL_ID = "amazon.titan-text-premier-v1:0"

def lambda_handler(event, context):
    patient_id = event.get("patientId")
    query = event.get("query")
    doctor_id = event.get("doctor_id")
    patient_info = event.get("patient_info")
    kb_response = event.get("kb_response")

    user_is_doctor = bool(doctor_id)

    final_answer = generate_final_response(query, patient_info, kb_response, patient_mode=not user_is_doctor)

    return {
        "patientId": patient_id,
        "query": query,
        "doctor_id": doctor_id,
        "final_answer": final_answer
    }

# ---------------- HELPERS ----------------

def generate_final_response(query, patient_info=None, kb_response=None, patient_mode=False):
    context_text = ""

    if patient_info:
        context_text += f"\nPatient Info:\n{json.dumps(patient_info)}"

    if kb_response and not patient_mode:
        context_text += f"\nKnowledge Base Response:\n{kb_response}"

    if patient_mode and kb_response:
        # KB contains polite denial message for patients
        context_text += f"\nNote:\n{kb_response.get('message', '')}"

    prompt = f"""
You are a professional medical assistant.  

User query: {query}
Content : {context_text}

Based on the content and query, provide a accurate, concise and professional short answer. 
Don't output anything unrelevent to the query. Dont need to address or greet the user. Only use the information given. 
If the information is NOT given, respond: "Insuddicient context."
"""

    body = {
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 500, "temperature": 0.2, "topP": 0.9},
    }

    response = bedrock.invoke_model(
        modelId=MODEL_FINAL_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    output = json.loads(response["body"].read())
    return output["results"][0]["outputText"].strip()