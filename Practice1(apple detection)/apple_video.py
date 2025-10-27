
import cv2
import numpy as np

# Open webcam (0 = default camera)
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(r"D:\Infosys_project\Python testing\apple.mp4")
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better display
    max_width = 800
    scale = max_width / frame.shape[1]
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red apple color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create mask for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Clean the mask (remove noise)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    apple_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            apple_count += 1

            # Get minimum enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Draw circle and number
            cv2.circle(frame, center, radius, (0, 255, 0), 3)
            cv2.putText(frame, f"{apple_count}", (int(x)-10, int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # Display total apple count
    cv2.putText(frame, f"Total Apples: {apple_count}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    # Show the live detection
    cv2.imshow("Live Apple Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
