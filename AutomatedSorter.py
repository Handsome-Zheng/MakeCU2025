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
... (237 lines left)
