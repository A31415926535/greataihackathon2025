import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

TABLE_NAME = "Medai_patientinfo"
MODEL_KB_ID = "amazon.titan-text-express-v1"

def lambda_handler(event, context):
    patient_id = event.get("patientId")
    query = event.get("query")
    doctor_id = event.get("doctor_id")
    classification = event.get("classification")

    user_is_doctor = bool(doctor_id)

    # Step 1: Fetch patient info if needed
    patient_info = None
    if classification in ["dynamo", "both"]:
        patient_info = get_patient_from_dynamo(patient_id)

    # Step 2: Fetch KB info if allowed
    kb_response = None
    if classification in ["kb", "both"]:
        if user_is_doctor:
            kb_response = call_bedrock(query, patient_info if classification == "both" else None, doctor=True)
        else:
            # Patient cannot access KB
            kb_response = {
                "message": "We are unable to provide clinical guideline information due to hospital privacy rules."
            }

    # Output for Lambda3
    return {
        "patientId": patient_id,
        "query": query,
        "doctor_id": doctor_id,
        "classification": classification,
        "patient_info": patient_info,
        "kb_response": kb_response
    }

# ---------------- HELPERS ----------------

def get_patient_from_dynamo(patient_id: str):
    table = dynamodb.Table(TABLE_NAME)
    resp = table.get_item(Key={"patient_id": patient_id})
    return resp.get("Item", {"message": f"No patient found with ID {patient_id}"})

def call_bedrock(query: str, patient_info=None, doctor=True):
    if doctor:
        prompt = f"""
You are a medical knowledge assistant.

- Do NOT answer the query directly.
- Your job is to understand what clinical guidelines or medical knowledge are relevant.
- If unsure about the guidelines, respond: "We are not sure".
- Output JSON with relevant KB data only.
Query: {query}
Patient Info: {json.dumps(patient_info) if patient_info else "None"}
"""
    else:
        prompt = f"""
You are a medical knowledge assistant.

- You are NOT allowed to provide clinical guidelines.
- Only patient-specific info can be returned.
- If the query requests guidelines, respond politely:
  "We are unable to provide clinical guideline information for patients due to hospital privacy rules."
Query: {query}
"""

    body = {
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 500, "temperature": 0.2, "topP": 0.9},
    }

    response = bedrock.invoke_model(
        modelId=MODEL_KB_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    output = json.loads(response["body"].read())
    return output["results"][0]["outputText"].strip()
