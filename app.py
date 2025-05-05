from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import io
import os
import cv2
from typing import Tuple

app = FastAPI(title="Rice Paddy Disease Detection API",
              description="API for detecting bacterial infections in rice paddy leaves")

# Allow CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)       

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold

# Load model once at startup
MODEL_PATH = "best_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Define classes
CLASSES = ['Bacteria', 'Healthy Rice Leaf']

def is_allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(content: bytes) -> bool:
    """Check if file size is within limits"""
    return len(content) <= MAX_FILE_SIZE
def is_plant_image(image: Image.Image) -> bool:
    """Improved plant detection with less restrictive color ranges"""
    try:
        # Convert to HSV color space
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define broader green-yellow color ranges (for healthy and diseased plants)
        lower_green = np.array([25, 25, 25])  # More permissive lower bound
        upper_green = np.array([95, 255, 255])  # More permissive upper bound
        
        # Calculate percentage of plant-like pixels
        mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.sum(mask > 0) / (img_array.shape[0] * img_array.shape[1])
        
        # Lower threshold for acceptance
        return green_percentage > 0.15
    except Exception as e:
        print(f"Plant detection error: {str(e)}")
        return False  # If there's an error, don't reject the image
def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize to model input size
        img = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Apply MobileNetV2 preprocessing
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to preprocess image: {str(e)}")
  

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
def validate_image_quality(img_array):
    """Validate image quality with more permissive thresholds"""
    if len(img_array.shape) == 4:
        # Extract the first image from batch
        img = img_array[0]
        
        # Convert to 0-255 range
        img_for_quality = ((img + 1) * 127.5).astype(np.uint8)
        
        # Check if image is extremely dark or bright (more permissive)
        mean_pixel = np.mean(img_for_quality)
        if mean_pixel < 20 or mean_pixel > 235:  # More permissive thresholds
            return False, "Image is too dark or too bright"
        
        # Check for severe blur (more permissive)
        gray = np.dot(img_for_quality[..., :3], [0.299, 0.587, 0.114])
        variance = np.var(gray)
        if variance < 300:  # Lower threshold to allow slightly blurrier images
            return False, "Image appears to be severely blurry"
            
        return True, "OK"
    else:
        raise ValueError("Expected preprocessed image with batch dimension")
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint for rice leaf disease prediction"""
    # File type check
    if not file.content_type.startswith("image/"):
        print(f"Rejected: Invalid content type {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are accepted.")
    
    contents = await file.read()

    try:
        # Validate file type
        if not is_allowed_file(file.filename):
            print(f"Rejected: Invalid file extension for {file.filename}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size
        if not validate_file_size(contents):
            print(f"Rejected: File too large ({len(contents)//1024//1024}MB)")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max size: {MAX_FILE_SIZE//1024//1024}MB"
            )
        
        # Preprocess image with validation
        print("Processing image...")
        img_array = preprocess_image(contents)

        # Validate image quality
        print("Checking image quality...")
        quality_ok, message = validate_image_quality(img_array)
        if not quality_ok:
            print(f"Rejected: Image quality issue - {message}")
            raise HTTPException(status_code=400, detail=f"Image quality issue: {message}")

        # Open original image for plant detection
        print("Checking if image contains plant foliage...")
        original_img = Image.open(io.BytesIO(contents)).convert('RGB')
        if not is_plant_image(original_img):
            print("Rejected: No plant foliage detected")
            raise HTTPException(
                status_code=400,
                detail="The image doesn't appear to contain plant foliage"
            )
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array)[0]
        predicted_index = np.argmax(prediction)
        predicted_class = CLASSES[predicted_index]
        confidence = float(np.max(prediction))
        
        # Validate confidence level
        if confidence < MIN_CONFIDENCE:
            print(f"Rejected: Low confidence ({confidence*100:.1f}%)")
            raise HTTPException(
                status_code=400,
                detail=f"Low confidence ({confidence*100:.1f}%). Please provide a clearer image of a rice leaf."
            )
        
        print(f"Prediction successful: {predicted_class} with {confidence*100:.1f}% confidence")
        return JSONResponse(content={
            "class": predicted_class,
            "confidence": round(confidence * 100, 2),
            "description": "Bacterial infection detected" if predicted_class == "Bacteria" 
                          else "No bacterial infection detected"
        })
        
    except HTTPException:
        raise  # Re-raise our own exceptions
    except ValueError as e:
        print(f"ValueError: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )
    
if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)