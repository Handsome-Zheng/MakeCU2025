#!/usr/bin/env python3
"""
Interactive Servo Calibration Script
Find the exact center, min, and max positions for your servo
"""
from gpiozero import Servo
import time
import sys

SERVO_PIN = 18

print("="*60)
print("           SERVO CALIBRATION TOOL")
print("="*60)
print(f"\nServo on GPIO {SERVO_PIN}")
print("\nThis tool helps you find:")
print("  1. Center position (0°)")
print("  2. Left position (-180°)")
print("  3. Right position (+180°)")
print("\n" + "="*60)

try:
    servo = Servo(SERVO_PIN)
    print("\n✓ Servo initialized\n")

    # Start at middle
    servo.value = 0
    print("Servo moved to default center (0)\n")
    time.sleep(1)

    # STEP 1: Find center
    print("="*60)
    print("STEP 1: CALIBRATE CENTER POSITION")
    print("="*60)
    print("\nCommands:")
    print("  +  : Move right (increase by 0.05)")
    print("  -  : Move left (decrease by 0.05)")
    print("  ++ : Move right (increase by 0.01)")
    print("  -- : Move left (decrease by 0.01)")
    print("  c  : Confirm this as center")
    print("  q  : Quit\n")

    current_pos = 0.0

    while True:
        print(f"Current position: {current_pos:+.3f} ", end='')
        cmd = input("Enter command: ").strip().lower()

        if cmd == '+':
            current_pos += 0.05
        elif cmd == '++':
            current_pos += 0.01
        elif cmd == '-':
            current_pos -= 0.05
        elif cmd == '--':
            current_pos -= 0.01
        elif cmd == 'c':
            center_pos = current_pos
            print(f"\n✓ Center position saved: {center_pos:+.3f}\n")
            break
        elif cmd == 'q':
            print("\nCalibration cancelled")
            sys.exit(0)
        else:
            print("Invalid command!")
            continue

        # Clamp to valid range
        current_pos = max(-1.0, min(1.0, current_pos))
        servo.value = current_pos
        time.sleep(0.1)

    # STEP 2: Find left (-180°)
    print("\n" + "="*60)
    print("STEP 2: CALIBRATE LEFT POSITION (-180°)")
    print("="*60)
    print("\nMove servo to full LEFT position")
    print("(The position for TRASH bin)\n")

    servo.value = -1.0
    current_pos = -1.0
    time.sleep(1)

    while True:
        print(f"Current position: {current_pos:+.3f} ", end='')
        cmd = input("Enter command: ").strip().lower()

        if cmd == '+':
            current_pos += 0.05
        elif cmd == '++':
            current_pos += 0.01
        elif cmd == '-':
            current_pos -= 0.05
        elif cmd == '--':
            current_pos -= 0.01
        elif cmd == 'c':
            left_pos = current_pos
            print(f"\n✓ Left position saved: {left_pos:+.3f}\n")
            break
        elif cmd == 'q':
            print("\nCalibration cancelled")
            sys.exit(0)
        else:
            print("Invalid command!")
            continue

        current_pos = max(-1.0, min(1.0, current_pos))
        servo.value = current_pos
        time.sleep(0.1)

    # STEP 3: Find right (+180°)
    print("\n" + "="*60)
    print("STEP 3: CALIBRATE RIGHT POSITION (+180°)")
    print("="*60)
    print("\nMove servo to full RIGHT position")
    print("(The position for RECYCLE bin)\n")

    servo.value = 1.0
    current_pos = 1.0
    time.sleep(1)

    while True:
        print(f"Current position: {current_pos:+.3f} ", end='')
        cmd = input("Enter command: ").strip().lower()

        if cmd == '+':
            current_pos += 0.05
        elif cmd == '++':
            current_pos += 0.01
        elif cmd == '-':
            current_pos -= 0.05
        elif cmd == '--':
            current_pos -= 0.01
        elif cmd == 'c':
            right_pos = current_pos
            print(f"\n✓ Right position saved: {right_pos:+.3f}\n")
            break
        elif cmd == 'q':
            print("\nCalibration cancelled")
            sys.exit(0)
        else:
            print("Invalid command!")
            continue

        current_pos = max(-1.0, min(1.0, current_pos))
        servo.value = current_pos
        time.sleep(0.1)

    # Test sequence
    print("\n" + "="*60)
    print("TESTING CALIBRATION")
    print("="*60)
    print("\nTesting movement sequence...\n")

    print("Moving to CENTER...")
    servo.value = center_pos
    time.sleep(2)

    print("Moving to LEFT (TRASH)...")
    servo.value = left_pos
    time.sleep(2)

    print("Moving to CENTER...")
    servo.value = center_pos
    time.sleep(2)

    print("Moving to RIGHT (RECYCLE)...")
    servo.value = right_pos
    time.sleep(2)

    print("Moving to CENTER...")
    servo.value = center_pos
    time.sleep(1)

    # Save results
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE!")
    print("="*60)
    print("\nYour calibrated values:")
    print(f"  SERVO_CENTER  = {center_pos:+.3f}")
    print(f"  SERVO_TRASH   = {left_pos:+.3f}   # Left position (-180°)")
    print(f"  SERVO_RECYCLE = {right_pos:+.3f}  # Right position (+180°)")
    print("\nUpdate these values in waste_sorter.py:")
    print("-" * 60)
    print(f"SERVO_CENTER = {center_pos}")
    print(f"SERVO_TRASH = {left_pos}")
    print(f"SERVO_RECYCLE = {right_pos}")
    print("-" * 60)

    # Save to file
    with open("servo_calibration.txt", "w") as f:
        f.write(f"# Servo Calibration Results\n")
        f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"SERVO_CENTER = {center_pos}\n")
        f.write(f"SERVO_TRASH = {left_pos}\n")
        f.write(f"SERVO_RECYCLE = {right_pos}\n")

    print("\n✓ Calibration saved to: servo_calibration.txt")
    print("\nGoodbye!")

except KeyboardInterrupt:
    print("\n\nCalibration cancelled")
    sys.exit(0)
except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
