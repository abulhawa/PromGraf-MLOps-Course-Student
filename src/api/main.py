import logging
import os
import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classifier API",
    description="API for classifying news articles into categories using a Hugging Face model.",
    version="1.0.0",
)

# --- Metrics definition ---
registry = CollectorRegistry()

# Counter 'api_requests_total', label by endpoint, method, and status code
api_requests_total = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

# Histogram 'api_request_duration_seconds', label by endpoint, method, and status code
api_request_duration_seconds = Histogram(
    "api_request_duration_seconds",
    "API request duration in seconds",
    ["endpoint", "method", "status_code"],
    registry=registry,
)

# Counter 'predictions_by_category'
predictions_by_category = Counter(
    "predictions_by_category",
    "Number of predictions by category",
    ["category"],
    registry=registry,
)


# Load the Hugging Face model
# This will download the model weights the first time it's run
try:
    classifier = pipeline(
        "text-classification", model="dima806/news-category-classifier-distilbert"
    )
    logger.info(
        "Hugging Face model loaded successfully: dima806/news-category-classifier-distilbert"
    )
except Exception as e:
    logger.error(f"Error loading Hugging Face model: {e}")
    raise RuntimeError("Error: Failed to load ML model.") from e


class ArticleInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    category: str
    score: float


@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the News Classifier API. Use /predict to classify articles."
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(article: ArticleInput):
    """
    Classify a news article based on its text content.
    """
    start_time = time.perf_counter()
    if not article.text:
        logger.warning("Received empty text for prediction.")

        api_requests_total.labels(
            endpoint="/predict", method="POST", status_code=400
        ).inc()

        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code=400
        ).observe(time.perf_counter() - start_time)

        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        # Perform prediction
        results = classifier(article.text)
        if not results:
            logger.error(
                f"Classifier returned empty results for text: {article.text[:50]}..."
            )
            raise HTTPException(
                status_code=500, detail="Model could not classify the text."
            )

        # Assuming the first result is the most confident one
        predicted_category = results[0]["label"]
        confidence_score = results[0]["score"]

        logger.info(
            f"Classified text (first 50 chars): '{article.text[:50]}...' into category: '{predicted_category}' with score: {confidence_score:.4f}"
        )
        prediction = PredictionOutput(
            category=predicted_category, score=confidence_score
        )
        api_requests_total.labels(
            endpoint="/predict", method="POST", status_code=200
        ).inc()
        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code=200
        ).observe(time.perf_counter() - start_time)

        predictions_by_category.labels(category=predicted_category).inc()

        return prediction

    except Exception as e:
        logger.error(
            f"Error during prediction for text: {article.text[:50]}... Error: {e}"
        )
        api_requests_total.labels(
            endpoint="/predict", method="POST", status_code=500
        ).inc()
        api_request_duration_seconds.labels(
            endpoint="/predict", method="POST", status_code=500
        ).observe(time.perf_counter() - start_time)
        raise HTTPException(
            status_code=500, detail=f"Prediction failed due to an internal error: {e}"
        )


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(registry), media_type="text/plain")
