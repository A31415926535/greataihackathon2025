import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
MODEL1_ID = "amazon.titan-text-express-v1"

def lambda_handler(event, context):
    """
    Input JSON:
    {
        "patientId": "...",
        "query": "...",
        "doctor_id": "..."  # optional
    }
    """
    try:
        patient_id = event.get("patientId")
        query = event.get("query")
        doctor_id = event.get("doctor_id")

        if not patient_id or not query:
            return {"statusCode": 400, "body": json.dumps({"error": "Missing patientId or query"})}

        classification = classify_query_ai(query)

        # Output JSON standardized for Lambda2
        return {
            "patientId": patient_id,
            "query": query,
            "doctor_id": doctor_id,
            "classification": classification
        }

    except Exception as e:
        logger.error(str(e))
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}

# ---------------- HELPER ----------------
def classify_query_ai(query: str) -> str:
    prompt = (
        "Classify the following medical query into one of these categories:\n"
        "1. dynamo - patient-specific data only (allergies, blood type, history).\n"
        "2. kb - general guidelines or medical knowledge.\n"
        "3. both - if it requires both patient data and general knowledge.\n\n"
        f"Query: {query}\n"
        "Answer with only: dynamo, kb, or both."
    )

    body = {
        "inputText": prompt,
        "textGenerationConfig": {"maxTokenCount": 20, "temperature": 0.0, "topP": 1.0},
    }

    response = bedrock.invoke_model(
        modelId=MODEL1_ID,
        body=json.dumps(body),
        accept="application/json",
        contentType="application/json",
    )

    output = json.loads(response["body"].read())
    classification = output["results"][0]["outputText"].strip().lower()

    if classification not in ["dynamo", "kb", "both"]:
        classification = "both"

    return classification