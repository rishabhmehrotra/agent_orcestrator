from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import logging
import time
import json
import os

# Ensure the log directory exists
log_directory = "/var/log/fastapi"
os.makedirs(log_directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(f"{log_directory}/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Check if CUDA is available and set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Load your models and label encoder
model_filenames = {
    "intentv1": "../trained-models/intent_classifier_Logistic Regression_Embedding1.pkl",
    "linearv1": "../trained-models/linear_issue_classifier.pkl",
    "linearv2": "../trained-models/linear_issue_classifier_v2.pkl",
    "highlevelv1": "../trained-models/high_level_question_classifier.pkl",
    "multi_class_intent": "../trained-models/multi_class_intent_classifier.pkl"  # New model
}

models = {}
label_encoder = None
try:
    for name, filename in model_filenames.items():
        models[name] = joblib.load(filename)
    if "multi_class_intent" in model_filenames:
        label_encoder = joblib.load('../trained-models/multi_class_label_encoder.pkl')
    logger.info("Models and label encoder loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models or label encoder: {e}")
    raise e

# Load the tokenizer and embedding model
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
logger.info(f"Tokenizer and embedding model loaded successfully from {embedding_model_name}")

class PredictionRequest(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    query: str
    intent: str
    score: float

app = FastAPI()

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "process_time": f"{process_time:.2f}s"
    }))
    return response

def extract_embedding(text):
    try:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Move tensor to CPU before converting to numpy
        return embeddings.flatten()
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        raise e

def predict_model(model, embedding, model_name):
    prediction_scores = model.predict_proba(embedding)
    
    if model_name == "intentv1":
        prediction = model.predict(embedding)
        intent_name = label_encoder.inverse_transform(prediction)[0]
        score = np.max(prediction_scores)
    elif model_name == "linearv1" or model_name == "linearv2":
        score_linear_issue = prediction_scores[0][1]
        if score_linear_issue > 0.9:
            intent_name = "Fix Linear Issue"
        else:
            intent_name = "General Query"
        score = score_linear_issue
    elif model_name == "highlevelv1":
        score_high_level = prediction_scores[0][1]
        if score_high_level > 0.4:
            intent_name = "High-Level Question"
        else:
            intent_name = "NOT high-level Question"
        score = score_high_level
    elif model_name == "multi_class_intent":
        prediction = model.predict(embedding)
        intent_name = label_encoder.inverse_transform(prediction)[0]
        score = np.max(prediction_scores)
    
    return intent_name, score

@app.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(model_name: str, request: PredictionRequest):
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    try:
        logger.info(f"Received query: {request.query} for model: {model_name}")
        embedding = extract_embedding(request.query).reshape(1, -1)
        intent_name, score = predict_model(models[model_name], embedding, model_name)
        logger.info(json.dumps({
            "query": request.query,
            "model": model_name,
            "prediction": intent_name,
            "score": score
        }))
        return PredictionResponse(query=request.query, intent=intent_name, score=score)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
