import streamlit as st
import json
import time
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta

# Streamlit app configuration - must be the first Streamlit command
st.set_page_config(
    page_title="Prescription Analysis System", 
    page_icon="📋", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configurable settings
CACHE_EXPIRY_MINUTES = 60
API_RETRY_SECONDS = 300
MODEL_PATH = '3rdpresc.pt'

# API key for Gemini
GEMINI_API_KEY = "YOUR_API_KEY"

# Import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    st.warning("Ultralytics YOLO package not found. Install with: pip install ultralytics")

# Import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    # Configure Gemini with the API key
    genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    GEMINI_AVAILABLE = False
    st.warning("Google Generative AI package not found. Install with: pip install google-generativeai")

# Cache setup
def get_cache_key(image):
    """Generate consistent hash key for image"""
    return hashlib.md5(image.tobytes()).hexdigest()

def get_cached_response(cache_key):
    """Retrieve cached response if available and not expired"""
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
        
    cache = st.session_state["analysis_cache"]
    if cache_key in cache:
        entry = cache[cache_key]
        if datetime.now() - entry["timestamp"] < timedelta(minutes=CACHE_EXPIRY_MINUTES):
            return entry["response"]
    return None

def store_cached_response(cache_key, response):
    """Store response in cache"""
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}
    st.session_state["analysis_cache"][cache_key] = {
        "response": response,
        "timestamp": datetime.now()
    }

# Configure Google Gemini API
def setup_gemini():
    if not GEMINI_AVAILABLE:
        return None
    try:
        return genai.GenerativeModel('gemini-1.5-pro-latest')
    except Exception as e:
        st.error(f"Error setting up Gemini: {str(e)}")
        return None

# Load the YOLO model with error handling
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    if not YOLO_AVAILABLE:
        return None
        
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Function to extract text from image using OCR (placeholder)
def extract_text_from_image(image):
    """Placeholder for OCR functionality"""
    return "Prescription text would appear here", []

# Function to process results with Gemini
def process_with_gemini(gemini_model, detection_data, extracted_text, detected_medicines):
    """Send detection data to Gemini API and process the response"""
    if not gemini_model:
        return {"error": "Gemini model not available"}

    # Retry mechanism
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create a list of detected medicines with their confidence scores
            medicine_list = []
            for item in detection_data:
                medicine_list.append({
                    "name": item["name"],
                    "confidence": item["confidence"]
                })
            
            prompt = f"""
            Analyze this prescription data:
            
            Detection Data:
            {json.dumps(detection_data, indent=2)}
            
            Detected Medicines:
            {json.dumps(medicine_list, indent=2)}
            
            Extracted Text:
            {extracted_text}
            
            Tasks:
            1. Validate detected medicine names in the data
            2. Correct any errors in medication names
            3. Extract dosage information
            4. Return JSON with structure:
            {{
                "medicines": [
                    {{
                        "detected": "EXACT_ORIGINAL_DETECTED_NAME",
                        "validated": "corrected name",
                        "dosage": "dosage info",
                        "confidence": 0-1,
                        "valid": true/false
                    }}
                ],
                "summary": "analysis summary"
            }}
            
            IMPORTANT INSTRUCTIONS:
            - You MUST use the exact detected name from the Detection Data in the "detected" field
            - The "detected" field must match exactly with the "name" field in the Detection Data
            - Do not substitute "MedicineName" or any generic placeholder for detected names
            - Each medicine in your response should correspond to an actual detected item
            - If unsure about any medicine, mark it as valid=false
            """

            # Generate content with safety settings
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
                }
            ]
            
            response = gemini_model.generate_content(
                prompt,
                safety_settings=safety_settings,
                generation_config={"temperature": 0.2}
            )
            
            # Handle API errors
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                return {"error": f"Content blocked: {response.prompt_feedback.block_reason}"}
            
            try:
                # Parse JSON response
                response_text = response.text
                result = json.loads(response_text)
                
                # Validate the response structure
                if "medicines" not in result:
                    return {"error": "Invalid response format: missing 'medicines' field", "text_response": response_text}
                
                # Ensure all medicines in the response match the detected names
                detected_names = {item["name"]: item for item in detection_data}
                
                # Validate medicines in the response
                for i, med in enumerate(result["medicines"]):
                    # Check if detected name is in our original detection data
                    if "detected" not in med or med["detected"] not in detected_names:
                        # If not found, try to match it with an existing detected name
                        if i < len(detected_medicines):
                            result["medicines"][i]["detected"] = detected_medicines[i]
                        else:
                            # Last resort fallback
                            result["medicines"][i]["detected"] = list(detected_names.keys())[0] if detected_names else "Unknown"
                
                # Add the original detected names to the response
                result["original_detected_names"] = detected_medicines
                
                return result
                
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                return {"text_response": response_text}

        except Exception as e:
            error_message = str(e).lower()
            if "quota" in error_message or "429" in error_message:
                return {
                    "error": "API quota exceeded. Please try again later.",
                    "retry_after": API_RETRY_SECONDS
                }
            elif "timeout" in error_message or "deadline" in error_message:
                # Retry on timeout
                retry_count += 1
                if retry_count < max_retries:
                    st.warning(f"Request timed out. Retrying ({retry_count}/{max_retries})...")
                    time.sleep(2)  # Wait before retrying
                    continue
                else:
                    return {"error": "API request timed out after multiple attempts"}
            else:
                return {"error": f"API error: {str(e)}"}
    
    return {"error": "Failed to process with Gemini API after multiple attempts"}



# App title and description
st.title("📋 Prescription Analysis System")
st.write("Upload a prescription image to analyze its contents and create structured data tables.")

# Initialize model
model = load_model(MODEL_PATH)

# Upload image section
st.subheader("Upload Prescription")
uploaded_file = st.file_uploader(
    "Choose a prescription image...",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of the prescription for analysis"
)

# Sidebar configuration options
st.sidebar.subheader("Configuration")

# Check if model loaded successfully
if not YOLO_AVAILABLE:
    st.error("⚠️ YOLO package not installed. Please install with: pip install ultralytics")
elif model is None:
    st.error("⚠️ Model not loaded. Please check the model file.")
else:
    if uploaded_file:
        try:
            # Process image
            image = Image.open(uploaded_file)
            cache_key = get_cache_key(image)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(image, caption='Uploaded Prescription', use_column_width=True)

            # Check cache first
            cached_result = get_cached_response(cache_key)
            if cached_result:
                st.info("Showing cached results from previous analysis")
                analysis = cached_result
            else:
                # Perform detection
                with st.spinner("Analyzing prescription..."):
                    try:
                        results = model(image)
                        
                        with col2:
                            st.success("Detection complete!")
                            if results and len(results) > 0:
                                res_image = results[0].plot()
                                st.image(res_image, caption='Detected Items', use_container_width=True)
                            else:
                                st.warning("No detection results available")

                        # Prepare detection data
                        detection_data = []
                        detected_item_names = []
                        
                        # Ensure results exists and has entries
                        if results and len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                            for i, box in enumerate(results[0].boxes):
                                if hasattr(box, 'cls') and hasattr(box, 'conf') and hasattr(box, 'xyxy'):
                                    class_id = int(box.cls[0]) if hasattr(box.cls, '__iter__') else int(box.cls)
                                    # Validate class_id is valid for model.names dictionary
                                    if class_id in model.names:
                                        class_name = model.names[class_id]
                                        confidence = float(box.conf[0]) if hasattr(box.conf, '__iter__') else float(box.conf)
                                        coords = box.xyxy[0].tolist() if hasattr(box.xyxy[0], 'tolist') else box.xyxy[0]
                                        
                                        detection_data.append({
                                            "id": i + 1,
                                            "name": class_name,
                                            "confidence": confidence,
                                            "bbox": [int(coord) for coord in coords]
                                        })
                                        detected_item_names.append(class_name)
                        
                        if not detection_data:
                            st.warning("No items detected in the prescription image")
                            analysis = {"error": "No items detected"}
                        else:
                            # Extract text
                            extracted_text, _ = extract_text_from_image(image)

                            # Debug info
                            if st.sidebar.checkbox("Show Debug Info"):
                                st.sidebar.subheader("Detected Items")
                                st.sidebar.json(detection_data)
                                st.sidebar.subheader("Detected Names")
                                st.sidebar.write(detected_item_names)

                            # Process with Gemini API
                            gemini_model = setup_gemini()
                            if gemini_model:
                                with st.spinner("Validating medicine information..."):
                                    analysis = process_with_gemini(gemini_model, detection_data, extracted_text, detected_item_names)
                                    # Only cache successful analyses
                                    if "error" not in analysis:
                                        store_cached_response(cache_key, analysis)
                            else:
                                analysis = {"error": "Gemini model not available"}

                    except Exception as e:
                        st.error(f"Error during detection: {str(e)}")
                        analysis = {"error": f"Processing error: {str(e)}"}

            # Display results
            if "error" in analysis:
                error_msg = analysis["error"]
                if "quota" in error_msg.lower() or "429" in error_msg:
                    st.error("⚠️ API Limit Reached")
                    st.markdown(f"""
                    **We've hit the API rate limits.** Here's what you can do:
                    - Wait and try again later (recommended wait: {analysis.get('retry_after', API_RETRY_SECONDS)} seconds)
                    - [Check your quota usage](https://makersuite.google.com/app/quotas)
                    - [Upgrade your plan](https://ai.google.dev/pricing)
                    """)
                else:
                    st.error(f"Analysis error: {error_msg}")
            elif "medicines" in analysis:
                
                # Display validated results
                st.subheader("Medicines Analysis")
                
                # Create and display results table
                results_df = pd.DataFrame(analysis["medicines"])
                
                # Format confidence values
                if 'confidence' in results_df.columns:
                    results_df['confidence'] = results_df['confidence'].apply(lambda x: f"{float(x):.2f}")
                
                # Define highlighting function for valid/invalid rows
                def highlight_valid(row):
                    if 'valid' in row and pd.notna(row['valid']):
                        color = 'lightgreen' if row['valid'] else 'lightcoral'
                        return ['background-color: ' + color] * len(row)
                    return [''] * len(row)
                
                # Show dataframe with column configuration
                try:
                    st.dataframe(
                        results_df.style.apply(highlight_valid, axis=1),
                        column_config={
                            "detected": "Detected Name",
                            "validated": "Validated Name",
                            "dosage": "Dosage",
                            "confidence": "Confidence",
                            "valid": "Is Valid"
                        },
                        use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Error displaying results table: {str(e)}")
                    st.dataframe(results_df)  # Fallback to simple display

                # Show detailed information
                st.subheader("Detailed Analysis")
                for med in analysis["medicines"]:
                    if med.get('valid', False):
                        with st.expander(f"✅ {med.get('validated', med.get('detected', 'Unknown'))}", expanded=False):
                            st.write(f"**Detected as:** {med.get('detected', 'Unknown')}")
                            st.write(f"**Dosage:** {med.get('dosage', 'Not specified')}")
                            st.write(f"**Confidence:** {med.get('confidence', 'Unknown')}")
                    else:
                        with st.expander(f"❌ {med.get('detected', 'Unknown')}", expanded=False):
                            st.error("Could not validate this medicine")
                            st.write(f"**Suggested name:** {med.get('validated', 'Unknown')}")
                
                # Show summary
                if "summary" in analysis:
                    with st.expander("Analysis Summary"):
                        st.write(analysis["summary"])

                # Export options
                st.subheader("Export Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "Download CSV",
                        data=results_df.to_csv(index=False).encode('utf-8'),
                        file_name="prescription_analysis.csv",
                        mime="text/csv"
                    )
                with col2:
                    st.download_button(
                        "Download JSON",
                        data=json.dumps(analysis, indent=2),
                        file_name="prescription_analysis.json",
                        mime="application/json"
                    )
            elif "text_response" in analysis:
                st.info("Received text response:")
                st.write(analysis["text_response"])
            else:
                st.warning("No valid analysis results.")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    else:
        # Show a placeholder when no image is uploaded
        st.info("Please upload a prescription image to begin analysis.")
