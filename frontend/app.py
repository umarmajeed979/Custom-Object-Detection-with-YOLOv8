"""
Streamlit UI for the YOLOv8 custom object detection API.

Talks to the FastAPI backend over HTTP — this file has no model logic
of its own (see docs/CODING_STANDARDS.md, rule 2: inference lives in
core/predictor.py only). The user uploads an image or takes a webcam
snapshot; either way it's POSTed to /predict, and the response is
rendered as the image with boxes + labels drawn on it, plus a table
of the raw detections underneath.
"""

import io
import os

import requests
import streamlit as st
from PIL import Image, ImageDraw

# In docker-compose, the frontend talks to the backend by service name,
# not localhost — override via the API_URL env var (see docker-compose.yml).
API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"  # matches the prefix app/__init__.py mounts routes.router under
BOX_COLOR = "#39FF14"
TEXT_COLOR = "#000000"

st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")


@st.cache_data(ttl=30)
def check_health() -> dict | None:
    """Ping the backend health endpoint; cached briefly to avoid hammering it."""
    try:
        response = requests.get(f"{API_URL}{API_PREFIX}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw bounding boxes and labels on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for detection in detections:
        box = detection["box"]
        label = f'{detection["label"]} {detection["confidence"]:.0%}'
        coords = (box["x1"], box["y1"], box["x2"], box["y2"])

        draw.rectangle(coords, outline=BOX_COLOR, width=3)
        text_bbox = draw.textbbox((box["x1"], box["y1"]), label)
        draw.rectangle(text_bbox, fill=BOX_COLOR)
        draw.text((box["x1"], box["y1"]), label, fill=TEXT_COLOR)

    return annotated


def run_prediction(image_bytes: bytes) -> dict | None:
    """POST image bytes to the backend /predict endpoint."""
    try:
        response = requests.post(
            f"{API_URL}{API_PREFIX}/predict",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        return None


def render_results(image: Image.Image, result: dict) -> None:
    """Render the annotated image and the detections table for one prediction."""
    st.image(draw_detections(image, result["detections"]), caption="Detections", use_container_width=True)
    st.caption(f"Inference time: {result['inference_time_ms']:.1f} ms — model {result['model_version']}")

    if not result["detections"]:
        st.info("No objects detected above the confidence threshold.")
        return

    st.subheader("Detections")
    st.dataframe(
        [
            {
                "Label": d["label"],
                "Confidence": f'{d["confidence"]:.1%}',
                "Box (x1, y1, x2, y2)": (
                    f'({d["box"]["x1"]:.0f}, {d["box"]["y1"]:.0f}, '
                    f'{d["box"]["x2"]:.0f}, {d["box"]["y2"]:.0f})'
                ),
            }
            for d in result["detections"]
        ],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.title("YOLOv8 Custom Object Detection")

    health = check_health()
    if health:
        st.caption(f"Model loaded — task: **{health['task']}**")
    else:
        st.warning("Backend not reachable — start the API before predicting.")

    source = st.radio("Image source", ["Upload", "Webcam"], horizontal=True)
    uploaded = (
        st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if source == "Upload"
        else st.camera_input("Take a photo")
    )

    if uploaded is None:
        st.info("Upload an image or take a photo to run detection.")
        return

    image = Image.open(uploaded).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")

    with st.spinner("Running detection..."):
        result = run_prediction(buffer.getvalue())

    if result is not None:
        render_results(image, result)


if __name__ == "__main__":
    main()