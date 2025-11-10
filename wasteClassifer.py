#!/usr/bin/env python3
"""
Raspberry Pi Inference Script - Optimized for real-time waste classification
Uses TorchScript model exported from wasteClassifier.py

Usage:
    # Single image inference
    python pi_inference.py image.jpg

    # Live camera inference
    python pi_inference.py
"""
import json
import torch
import numpy as np
import cv2
import sys
import time

# ==================== CONFIGURATION ====================
MODEL_PATH = "model.ts"
LABELS_PATH = "labels.json"
IMG_SIZE = 300

# ImageNet normalization (same as training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== PREPROCESSING ====================
def preprocess_image(img):
    """
    Preprocess BGR image from OpenCV to model input tensor
    Args:
        img: BGR image from cv2.imread() or camera
    Returns:
        torch tensor ready for inference
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0

    # Apply ImageNet normalization
    img_normalized = (img_float - MEAN) / STD

    # Convert to CHW format and add batch dimension
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))[None])

    return img_tensor

# ==================== MODEL LOADING ====================
print("Loading model and labels...")
try:
    labels = json.load(open(LABELS_PATH))
    print(f"Classes: {labels}")

    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()

    # Optimize for CPU inference
    torch.set_num_threads(4)  # Adjust based on your Pi model

    print("✓ Model loaded successfully")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Classes: {len(labels)}")

except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# ==================== INFERENCE FUNCTIONS ====================
def predict_image(image_path):
    """Single image prediction"""
    print(f"\nProcessing image: {image_path}")

    # Load and preprocess
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return

    x = preprocess_image(frame)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).numpy()[0]
    inference_time = (time.time() - start_time) * 1000

    # Get prediction
    pred_idx = int(probabilities.argmax())
    confidence = float(probabilities[pred_idx])

    # Display results
    print(f"\nPrediction: {labels[pred_idx]}")
    print(f"Confidence: {confidence*100:.2f}%")
    print(f"Inference time: {inference_time:.1f}ms")
    print("\nAll probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  {labels[i]}: {prob*100:.2f}%")

def camera_inference():
    """Live camera inference"""
    print("\n" + "="*50)
    print("Starting camera inference...")
    print("Press Ctrl+C to quit")
    print("="*50 + "\n")
    
    # Try picamera2 first (best for Pi Camera Module)
    use_picamera = False
    try:
        from picamera2 import Picamera2
        print("Using picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        )
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Camera warm-up
        use_picamera = True
        print("✓ Camera started with picamera2\n")
    except ImportError:
        print("picamera2 not available, trying OpenCV...")
        use_picamera = False
    except Exception as e:
        print(f"picamera2 failed: {e}")
        print("Trying OpenCV fallback...\n")
        use_picamera = False
    
    # OpenCV fallback
    if not use_picamera:
        for camera_idx in [0, 1, 2, 3]:
            cap = cv2.VideoCapture(camera_idx)
            if cap.isOpened():
                print(f"✓ Camera opened with OpenCV (index: {camera_idx})\n")
                break
        if not cap.isOpened():
            print("Error: Could not open camera")
            print("Install picamera2: sudo apt install -y python3-picamera2")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    fps_time = time.time()
    fps_counter = 0
    fps = 0
    
    try:
        while True:
            # Capture frame
            if use_picamera:
                frame = picam2.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
            
            # Preprocess and infer
            x = preprocess_image(frame)
            start_time = time.time()
            with torch.no_grad():
                logits = model(x)
                probabilities = torch.softmax(logits, dim=1).numpy()[0]
            inference_time = (time.time() - start_time) * 1000
            
            # Get prediction
            pred_idx = int(probabilities.argmax())
            confidence = probabilities[pred_idx]
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()
            
            # Print to terminal (since no display over SSH)
            print(f"\r{labels[pred_idx]}: {confidence*100:.1f}% | "
                  f"FPS: {fps:.1f} | Inference: {inference_time:.1f}ms    ",
                  end='', flush=True)
            
            time.sleep(0.05)  # Small delay
            
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        if use_picamera:
            picam2.stop()
        else:
            cap.release()
        print("\nCamera inference stopped")

# ==================== MAIN ====================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("WASTE CLASSIFIER - RASPBERRY PI INFERENCE")
    print("="*50)

    if len(sys.argv) > 1:
        # Single image mode
        predict_image(sys.argv[1])
    else:
        # Live camera mode
        camera_inference()
