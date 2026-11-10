import logging
import os
import time

from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from sklearn.metrics import precision_recall_fscore_support
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

model_accuracy_score = Gauge(
    "model_accuracy_score",
    "Current accuracy of the News Classifier model",
    registry=registry,
)


model_precision_score = Gauge(
    "model_precision_score",
    "Precision by category in the latest evaluation",
    ["category"],
    registry=registry,
)
model_recall_score = Gauge(
    "model_recall_score",
    "Recall by category in the latest evaluation",
    ["category"],
    registry=registry,
)
model_f1_score = Gauge(
    "model_f1_score",
    "F1 score by category in the latest evaluation",
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


# Data model for evaluation
class EvaluationItem(BaseModel):
    text: str
    true_label: str


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


@app.post("/evaluate")
async def evaluate_model(items: list[EvaluationItem]):
    """Evaluate the model against labeled articles and publish its accuracy."""
    start_time = time.perf_counter()
    if not items:
        raise HTTPException(status_code=400, detail="No items provided for evaluation.")

    correct_predictions = 0
    true_labels = []
    predicted_labels = []
    for item in items:
        try:
            predicted_label = classifier(item.text)[0]["label"]
            true_labels.append(item.true_label.lower())
            predicted_labels.append(predicted_label.lower())
            if predicted_label.lower() == item.true_label.lower():
                correct_predictions += 1
        except Exception:
            logger.exception("Failed to classify an evaluation item.")

    accuracy = correct_predictions / len(items)
    model_accuracy_score.set(accuracy)
    categories = sorted(set(true_labels) | set(predicted_labels))

    model_precision_score.clear()
    model_recall_score.clear()
    model_f1_score.clear()

    if categories:
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=categories,
            average=None,
            zero_division=0,
        )
        for category, precision_score, recall_score, f1_score in zip(
            categories, precision, recall, f1
        ):
            model_precision_score.labels(category=category).set(precision_score)
            model_recall_score.labels(category=category).set(recall_score)
            model_f1_score.labels(category=category).set(f1_score)
    logger.info("Model evaluated. Accuracy: %.4f", accuracy)

    api_requests_total.labels(
        endpoint="/evaluate", method="POST", status_code=200
    ).inc()
    api_request_duration_seconds.labels(
        endpoint="/evaluate", method="POST", status_code=200
    ).observe(time.perf_counter() - start_time)

    return {
        "message": "Model evaluation completed",
        "accuracy": accuracy,
        "evaluated_items": len(items),
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(registry), media_type="text/plain")
