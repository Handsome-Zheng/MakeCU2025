#!/usr/bin/env python3
"""
Automated Waste Sorter - Continuous Operation
- Ultrasonic sensor detects object
- Camera captures and classifies waste
- Ultra-smooth servo sorting motion
"""
import json
import torch
import numpy as np
import cv2
import time
import signal
import sys
from gpiozero import DistanceSensor, Servo, Device
from picamera2 import Picamera2

# ==================== CONFIGURATION ====================
MODEL_PATH = "model.ts"
LABELS_PATH = "labels.json"
IMG_SIZE = 300

# GPIO Pin Configuration
ULTRASONIC_TRIGGER_PIN = 23
ULTRASONIC_ECHO_PIN = 24
SERVO_PIN = 18

# Detection settings
DETECTION_DISTANCE = 2.0     # inches
INFERENCE_DURATION = 3.0     # seconds

# Servo positions (standard 180° servo: -1.0 to +1.0)
SERVO_CENTER = -0.250        # Your calibrated zero
SERVO_RECYCLE = 1.4          # Far right (90° from center)
SERVO_TRASH = -1.8

# Far left (90° from center)

# Smooth motion settings
UPDATE_HZ = 100              # Update rate for smooth motion
MOVEMENT_DURATION = 1.0      # Seconds for servo movement

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ==================== SMOOTH MOTION FUNCTIONS ====================
def ease_in_out(t):
    """Smooth acceleration/deceleration curve"""
    return t * t * (3 - 2 * t)

def smooth_move_servo(target_position, duration=MOVEMENT_DURATION):
    """
    Ultra-smooth servo movement with ease-in-out
    Args:
        target_position: Target servo position (-1.0 to +1.0)
        duration: Time in seconds for movement
    """
    # Clamp target to valid range
    target_position = max(-1.0, min(1.0, target_position))

    # Get current position
    current = servo.value if servo.value is not None else SERVO_CENTER

    steps = int(duration * UPDATE_HZ)
    dt = duration / steps

    for i in range(steps + 1):
        progress = ease_in_out(i / steps)
        position = current + (target_position - current) * progress
        # Ensure position stays within bounds
        position = max(-1.0, min(1.0, position))
        servo.value = position
        time.sleep(dt)

    servo.value = target_position

def cleanup_and_exit(signum=None, frame=None):
    """Clean shutdown - return to center position"""
    print("\n\n" + "="*60)
    print("  SHUTTING DOWN...")
    print("="*60)

    # Return to center smoothly
    print(f"  Returning to center ({SERVO_CENTER:+.3f})...")
    try:
        smooth_move_servo(SERVO_CENTER, duration=1.5)
        time.sleep(0.5)
    except Exception as e:
        print(f"  Warning during servo centering: {e}")
        try:
            servo.value = SERVO_CENTER
            time.sleep(1)
        except:
            pass

    # Stop camera
    try:
        picam2.stop()
        print("  ✓ Camera stopped")
    except Exception as e:
        print(f"  Warning stopping camera: {e}")

    # Cleanup GPIO - compatible with both factories
    try:
        servo.close()
        sensor.close()
        print("  ✓ GPIO cleaned up")
    except Exception as e:
        print(f"  Warning during GPIO cleanup: {e}")

    if hasattr(cleanup_and_exit, 'stats'):
        stats = cleanup_and_exit.stats
        print(f"\n  Final Statistics:")
        print(f"    RECYCLE: {stats['RECYCLE']}")
        print(f"    TRASH: {stats['TRASH']}")
        print(f"    Total: {sum(stats.values())}")

    print("\n  Goodbye!\n")
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ==================== HARDWARE SETUP ====================
print("="*60)
print("  AUTOMATED WASTE SORTER")
print("="*60)
print("\nInitializing hardware...")

# Initialize servo with standard 180° range
servo = Servo(
    SERVO_PIN,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    frame_width=20/1000
)
print(f"✓ Servo on GPIO {SERVO_PIN}")
print(f"  Position range: ±1.0 (180° total rotation)")

# Ultrasonic sensor
sensor = DistanceSensor(
    echo=ULTRASONIC_ECHO_PIN,
    trigger=ULTRASONIC_TRIGGER_PIN,
    max_distance=1.0
)
print(f"✓ Ultrasonic sensor (Trigger: GPIO{ULTRASONIC_TRIGGER_PIN}, Echo: GPIO{ULTRASONIC_ECHO_PIN})")

# Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
time.sleep(2)
print("✓ Camera initialized")

# Move to center position smoothly
print(f"\nCentering servo at {SERVO_CENTER:+.3f}...")
servo.value = 0.0
time.sleep(0.5)
smooth_move_servo(SERVO_CENTER, duration=1.5)
print("✓ Servo centered")

# ==================== MODEL LOADING ====================
print("\nLoading AI model...")
labels = json.load(open(LABELS_PATH))
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()
torch.set_num_threads(4)
print(f"✓ Model loaded: {labels}")

# ==================== PREPROCESSING ====================
def preprocess_image(img):
    """Preprocess image for model"""
    img_resized = np.array(img)
    if len(img_resized.shape) == 3 and img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]

    img_resized = cv2.resize(img_resized, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img_float = img_resized.astype(np.float32) / 255.0
    img_normalized = (img_float - MEAN) / STD
    img_tensor = torch.from_numpy(np.transpose(img_normalized, (2, 0, 1))[None])
    return img_tensor

# ==================== INFERENCE ====================
def run_inference_window():
    """Run inference for configured duration and aggregate results"""
    print("\n[INFERENCE STARTED]")

    start_time = time.time()
    predictions = []
    confidences = []
    frame_count = 0

    while time.time() - start_time < INFERENCE_DURATION:
        # Capture frame
        frame = picam2.capture_array()

        # Preprocess
        x = preprocess_image(frame)

        # Inference
        with torch.no_grad():
            logits = model(x)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]

        # Get prediction
        pred_idx = int(probabilities.argmax())
        confidence = float(probabilities[pred_idx])
        prediction = labels[pred_idx]

        predictions.append(prediction)
        confidences.append(confidence)
        frame_count += 1

        # Show progress
        elapsed = time.time() - start_time
        print(f"\r  Progress: {elapsed:.1f}s / {INFERENCE_DURATION:.0f}s - Current: {prediction} ({confidence*100:.1f}%)",
              end='', flush=True)

        time.sleep(0.1)

    print()

    # Determine final classification by majority vote
    recycle_count = predictions.count("RECYCLE")
    trash_count = predictions.count("TRASH")

    if recycle_count > trash_count:
        final_classification = "RECYCLE"
        avg_confidence = np.mean([c for p, c in zip(predictions, confidences) if p == "RECYCLE"])
    else:
        final_classification = "TRASH"
        avg_confidence = np.mean([c for p, c in zip(predictions, confidences) if p == "TRASH"])

    print(f"[INFERENCE COMPLETE]")
    print(f"  Frames analyzed: {frame_count}")
    print(f"  RECYCLE votes: {recycle_count}")
    print(f"  TRASH votes: {trash_count}")
    print(f"  Final: {final_classification} ({avg_confidence*100:.1f}%)")

    return final_classification, avg_confidence

# ==================== SORTING ====================
def sort_waste(classification):
    """Sort waste with ultra-smooth servo motion"""
    print(f"\n[SORTING: {classification}]")

    # Determine target position
    if classification == "RECYCLE":
        target = SERVO_RECYCLE
        print(f"  → Rotating to RECYCLE position ({target:+.3f})")
    else:
        target = SERVO_TRASH
        print(f"  → Rotating to TRASH position ({target:+.3f})")

    # Smooth movement to target
    smooth_move_servo(target, duration=MOVEMENT_DURATION)

    # Hold position
    print(f"  → Holding position (2 seconds)...")
    time.sleep(2.0)

    # Return to center smoothly
    print(f"  → Returning to center ({SERVO_CENTER:+.3f})")
    smooth_move_servo(SERVO_CENTER, duration=MOVEMENT_DURATION * 1.5)

    print(f"  ✓ Sorting complete!")

# ==================== MAIN LOOP ====================
def main():
    print("\n" + "="*60)
    print("  SYSTEM READY - CONTINUOUS OPERATION")
    print("="*60)
    print(f"  Detection distance: {DETECTION_DISTANCE} inches")
    print(f"  Inference duration: {INFERENCE_DURATION} seconds")
    print(f"  Servo positions:")
    print(f"    Center:  {SERVO_CENTER:+.3f}")
    print(f"    Recycle: {SERVO_RECYCLE:+.3f} (180° rotation)")
    print(f"    Trash:   {SERVO_TRASH:+.3f} (180° rotation)")
    print("\n  Place waste in front of sensor...")
    print("  Press Ctrl+C to stop")
    print("="*60 + "\n")

    items_sorted = {"RECYCLE": 0, "TRASH": 0}
    cleanup_and_exit.stats = items_sorted

    try:
        while True:
            # Read distance in inches
            distance_inches = sensor.distance * 39.3701

            # Check if object detected
            if distance_inches <= DETECTION_DISTANCE:
                print(f"\n{'='*60}")
                print(f"[OBJECT DETECTED] Distance: {distance_inches:.2f} inches")

                # Wait for object to stabilize
                time.sleep(0.5)

                # Run inference
                classification, confidence = run_inference_window()

                # Sort waste
                sort_waste(classification)

                # Update statistics
                items_sorted[classification] += 1
                print(f"\n[STATISTICS]")
                print(f"  RECYCLE: {items_sorted['RECYCLE']}")
                print(f"  TRASH:   {items_sorted['TRASH']}")
                print(f"  Total:   {sum(items_sorted.values())}")

                # Wait for object removal
                print("\n[WAITING] Remove object to continue...")

                while sensor.distance * 39.3701 <= DETECTION_DISTANCE + 1:
                    time.sleep(0.3)

                print("[READY] Waiting for next object\n")
                time.sleep(1)

            time.sleep(0.1)

    except KeyboardInterrupt:
        cleanup_and_exit()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        cleanup_and_exit()

if __name__ == "__main__":
    main()
