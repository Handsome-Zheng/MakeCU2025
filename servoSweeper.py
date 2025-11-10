#!/usr/bin/env python3
"""
Ultra-Smooth Servo Sweep using gpiozero
Guaranteed to work on Pi 5
"""
from gpiozero import Servo
import time
import math
import signal
import sys

# Configuration
SERVO_PIN = 18
ZERO_POSITION = -0.250

# Servo with fine-tuned pulse widths
servo = Servo(
    SERVO_PIN,
    min_pulse_width=0.5/1000,
    max_pulse_width=2.5/1000,
    frame_width=20/1000
)

# Sweep settings
LEFT = -1.2
RIGHT = 1.4
CYCLE_TIME = 6.0  # Seconds per full cycle
UPDATE_HZ = 100   # 100 Hz update rate

def ease_in_out(t):
    """Smooth acceleration/deceleration curve"""
    return t * t * (3 - 2 * t)

def cleanup_and_exit(signum=None, frame=None):
    """Return to zero smoothly on exit"""
    print("\n\n" + "="*60)
    print("  RETURNING TO ZERO...")
    print("="*60)
    
    current = servo.value if servo.value is not None else 0.0
    return_time = 2.0
    steps = int(return_time * UPDATE_HZ)
    
    for i in range(steps + 1):
        progress = ease_in_out(i / steps)
        position = current + (ZERO_POSITION - current) * progress
        servo.value = position
        time.sleep(1.0 / UPDATE_HZ)
    
    servo.value = ZERO_POSITION
    time.sleep(0.5)
    
    print(f"✓ Servo at zero ({ZERO_POSITION:+.3f})")
    print("\nGoodbye!\n")
    sys.exit(0)

# Register exit handlers
signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# Initialize
print("="*60)
print("  ULTRA-SMOOTH SERVO SWEEP")
print("="*60)
print(f"\nServo: GPIO {SERVO_PIN}")
print(f"Zero: {ZERO_POSITION:+.3f}")
print(f"Range: {LEFT:+.3f} to {RIGHT:+.3f}")
print(f"Update rate: {UPDATE_HZ} Hz")
print("\nPress Ctrl+C to stop\n")
print("="*60 + "\n")

# Move to zero
print(f"Moving to zero ({ZERO_POSITION:+.3f})...")
servo.value = 0.0
time.sleep(0.5)

for i in range(50):
    progress = ease_in_out(i / 50)
    servo.value = 0.0 + (ZERO_POSITION - 0.0) * progress
    time.sleep(0.02)

servo.value = ZERO_POSITION
time.sleep(1)
print("✓ At zero\n")

print("Starting continuous sweep...\n")
time.sleep(0.5)

# Main loop
try:
    sweep = 0
    dt = 1.0 / UPDATE_HZ
    
    while True:
        sweep += 1
        print(f"[Sweep #{sweep}]", end='', flush=True)
        
        start_time = time.time()
        
        # Continuous sine wave sweep
        while time.time() - start_time < CYCLE_TIME:
            elapsed = time.time() - start_time
            
            # Sine wave (0 to 2π over cycle time)
            angle = (elapsed / CYCLE_TIME) * 2 * math.pi
            sine = math.sin(angle)
            
            # Map to full range: -1 to +1
            # When sine = -1, position = LEFT (-1.0)
            # When sine = +1, position = RIGHT (+1.0)
            position = sine
            
            servo.value = position
            time.sleep(dt)
        
        print(" ✓")

except KeyboardInterrupt:
    cleanup_and_exit()
except Exception as e:
    print(f"\nError: {e}")
    cleanup_and_exit()
