import streamlit as st
from PIL import Image
from ultralytics import YOLO
import json
import pandas as pd
import hashlib
from datetime import datetime, timedelta
import os
from openai import OpenAI
import re

# Streamlit app configuration
st.set_page_config(
    page_title="💊 Prescription Medicine Name Correction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurable settings
CACHE_EXPIRY_MINUTES = 60
MODEL_PATH = os.environ.get('MODEL_PATH', '3rdpresc.pt')

# Initialize OpenAI client - INSECURE VERSION (FOR TESTING ONLY)
# Initialize OpenAI client - INSECURE VERSION (FOR TESTING ONLY)
# Initialize OpenAI client - INSECURE VERSION (FOR TESTING ONLY)
def initialize_openai_client():
    try:
        # **WARNING: DO NOT HARDCODE YOUR API KEY IN PRODUCTION CODE!**
        # This is highly insecure. Use environment variables instead.
        api_key = "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"  # Replace with your API key for testing

        if api_key == "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":
            st.warning("API key is still the placeholder. Please replace it for AI corrections to work.")
            return None
        elif api_key:
            client = OpenAI(api_key=api_key)
            # Verify the key works with a simple request
            try:
                client.models.list()
                return client
            except Exception as auth_error:
                st.error(f"API key verification failed: {str(auth_error)}")
                return None
        return None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {str(e)}")
        return None

# Initialize client
client = initialize_openai_client()
OPENAI_AVAILABLE = client is not None

# Cache setup for YOLO detection
def get_cache_key_detection(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def get_cached_detection(cache_key):
    cache = st.session_state.get("detection_cache", {})
    if cache_key in cache:
        entry = cache[cache_key]
        if datetime.now() - entry["timestamp"] < timedelta(minutes=CACHE_EXPIRY_MINUTES):
            return entry["detection_data"], entry["detected_items"]
    return None, None

def store_cached_detection(cache_key, detection_data, detected_items):
    if "detection_cache" not in st.session_state:
        st.session_state["detection_cache"] = {}
    st.session_state["detection_cache"][cache_key] = {
        "detection_data": detection_data,
        "detected_items": detected_items,
        "timestamp": datetime.now()
    }

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model(MODEL_PATH)

# Process YOLO output with OpenAI for correction
def correct_medicine_names_with_openai(detected_items):
    if not OPENAI_AVAILABLE or not detected_items:
        return detected_items, None

    numbered_medicines = "\n".join([f"{i+1}. {item}" for i, item in enumerate(detected_items)])

    prompt = f"""
    Correct any spelling errors in the following list of potential medicine names:
    {numbered_medicines}

    Return the corrected list of medicine names as a JSON array.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that corrects spelling errors in medicine names."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response.choices[0].message.content)
            corrected_medicines = result.get("corrected_medicines", detected_items)
            return corrected_medicines, "Spelling corrections applied."
        except json.JSONDecodeError:
            st.warning("Received invalid JSON for corrections. Returning original.")
            return detected_items, "Invalid JSON response for corrections."

    except Exception as e:
        st.error(f"AI correction failed: {str(e)}")
        return detected_items, f"AI correction failed: {str(e)}"

# App UI
st.title("💊 Prescription Medicine Name Correction")
st.write("Upload a prescription image for medicine detection and AI-powered spelling correction.")

uploaded_file = st.file_uploader("Choose prescription image...", type=["jpg", "jpeg", "png"])

if model is None:
    st.error("Model failed to load. Please check configuration.")
elif uploaded_file:
    try:
        image = Image.open(uploaded_file)
        cache_key = get_cache_key_detection(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Prescription', use_column_width=True)

        # Check cache first
        cached_data, cached_items = get_cached_detection(cache_key)
        if cached_data:
            st.info("Showing cached results")
            detection_data = cached_data
            detected_items = cached_items
            with col2:
                st.image(image, caption='Cached Detection', use_container_width=True)
        else:
            with st.spinner("Detecting medicine names..."):
                try:
                    results = model(image)
                    with col2:
                        st.image(results[0].plot(), caption='Detection Results', use_container_width=True)

                    detection_data = []
                    detected_items = []
                    if hasattr(results[0], 'boxes'):
                        for box in results[0].boxes:
                            if hasattr(box, 'cls') and box.cls is not None:
                                class_id = int(box.cls)
                                detection_data.append({
                                    "name": model.names[class_id],
                                    "confidence": float(box.conf),
                                    "bbox": box.xyxy[0].tolist()
                                })
                                detected_items.append(model.names[class_id])

                    store_cached_detection(cache_key, detection_data, detected_items)

                except Exception as e:
                    st.error(f"Detection failed: {str(e)}")

        if detected_items:
            st.subheader("Detected Medicine Names")
            detected_list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(detected_items)])
            st.code(detected_list_str, language=None)
            st.download_button(
                "Download",
                detected_list_str,
                "detected_medicines.txt",
                "text/plain"
            )

            if OPENAI_AVAILABLE:
                with st.spinner("Applying AI corrections..."):
                    corrected_medicines, correction_notes = correct_medicine_names_with_openai(detected_items)
                    st.subheader("With AI corrections:")
                    if corrected_medicines != detected_items:
                        corrected_list_str = "\n".join([f"{i+1}. {item}" for i, item in enumerate(corrected_medicines)])
                        st.code(corrected_list_str, language=None)
                        st.download_button(
                            "Download",
                            corrected_list_str,
                            "corrected_medicines.txt",
                            "text/plain"
                        )
                        if correction_notes:
                            st.info(correction_notes)
                    else:
                        st.info("No spelling corrections needed.")
            else:
                st.warning("AI correction unavailable - no valid API key found. Please enter your API key in the code.")

            st.subheader("Technical Detection Data")
            st.dataframe(pd.DataFrame(detection_data))
            st.download_button(
                "Download Raw Data",
                json.dumps(detection_data, indent=2),
                "detection_data.json"
            )

    except Exception as e:
        st.error(f"Processing error: {str(e)}")

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown(f"**Cache Status:** {len(st.session_state.get('detection_cache', {}))} items")

if st.sidebar.button("Clear Cache"):
    st.session_state["detection_cache"] = {}
    st.sidebar.success("Cache cleared")

st.sidebar.markdown("""
---
**System Information**
- Model: YOLO v8
- AI Correction: {"Enabled" if OPENAI_AVAILABLE else "Disabled"}
- Version: 2.4
""")