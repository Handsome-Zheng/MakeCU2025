#!/usr/bin/env python3
"""Raspberry Pi Camera Module 3 Test Script"""
import sys
import time

print("="*50)
print("RASPBERRY PI CAMERA MODULE 3 TEST")
print("="*50)

# Test 1: Check camera detection
print("\n[1] Checking camera detection...")
import subprocess
try:
    result = subprocess.run(['vcgencmd', 'get_camera'],
                          capture_output=True, text=True, timeout=5)
    print(f"Camera status: {result.stdout.strip()}")
    if "detected=1" not in result.stdout:
        print("⚠️  WARNING: Camera not detected!")
except Exception as e:
    print(f"Could not check camera status: {e}")

# Test 2: Try picamera2 (recommended for Camera Module 3)
print("\n[2] Testing with picamera2...")
try:
    from picamera2 import Picamera2
    from PIL import Image as PILImage

    print("✓ picamera2 is installed")
    print("Initializing camera...")

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    print("Warming up camera (2 seconds)...")
    time.sleep(2)

    print("Capturing test image...")
    image = picam2.capture_array()

    print(f"✓ Image captured: {image.shape} (height, width, channels)")
    print(f"  Resolution: {image.shape[1]}x{image.shape[0]}")

    # Save test image
    img = PILImage.fromarray(image)
    img.save("camera_test_picamera2.jpg")
    print("✓ Test image saved: camera_test_picamera2.jpg")

    # Test continuous capture
    print("\nTesting continuous capture (5 frames)...")
    for i in range(5):
        frame = picam2.capture_array()
        print(f"  Frame {i+1}: {frame.shape}")
        time.sleep(0.2)

    picam2.stop()

    print("\n" + "="*50)
    print("✓ PICAMERA2 TEST PASSED!")
    print("="*50)
    print("Your Camera Module 3 is working perfectly!")
    print("Check the saved image: camera_test_picamera2.jpg")
    sys.exit(0)

except ImportError:
    print("✗ picamera2 not installed")
    print("  Installing: sudo apt install -y python3-picamera2")
    print("  Then run this script again")
    print("\nTrying OpenCV fallback...")
except Exception as e:
    print(f"✗ picamera2 failed: {e}")
    print("\nTrying OpenCV fallback...")

# Test 3: OpenCV fallback
print("\n[3] Testing with OpenCV...")
try:
    import cv2
    print("✓ OpenCV is installed")

    for camera_idx in [0, -1, 1]:
        print(f"\nTrying camera index {camera_idx}...")
        cap = cv2.VideoCapture(camera_idx)

        if cap.isOpened():
            print(f"✓ Camera opened with index {camera_idx}")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  Resolution: {width}x{height}")

            ret, frame = cap.read()
            if ret:
                print(f"✓ Frame captured: {frame.shape}")
                cv2.imwrite("camera_test_opencv.jpg", frame)
                print("✓ Test image saved: camera_test_opencv.jpg")

                # Test continuous capture
                print("\nTesting continuous capture (5 frames)...")
                for i in range(5):
                    ret, frame = cap.read()
                    if ret:
                        print(f"  Frame {i+1}: {frame.shape}")
                    time.sleep(0.2)

                cap.release()

                print("\n" + "="*50)
                print("✓ OPENCV TEST PASSED!")
                print("="*50)
                print(f"Camera working! Use index: {camera_idx}")
                sys.exit(0)

            cap.release()

    print("\n✗ Could not capture frames from camera")

except ImportError:
    print("✗ OpenCV not installed yet")
except Exception as e:
    print(f"✗ OpenCV test failed: {e}")

print("\n" + "="*50)
print("TROUBLESHOOTING:")
print("="*50)
print("1. Enable camera: sudo raspi-config > Interface Options > Camera")
print("2. Install picamera2: sudo apt install -y python3-picamera2")
print("3. Check connection: vcgencmd get_camera")
print("4. Reboot: sudo reboot")


# To test camera, run: rpicam-still -o test_photo.jpg --width 640 --height 480
