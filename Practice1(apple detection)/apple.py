import cv2
import numpy as np

# Load the image
image = cv2.imread(r"D:\Infosys_project\Python testing\image.png")
if image is None:
    raise ValueError("Image not found. Check your file path.")

# ---- Resize image for better display ----
max_width = 800  # change as per your screen size
scale = max_width / image.shape[1]
image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red apple color range (you can adjust for green apples)
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
    if area > 500:  # Ignore small objects
        apple_count += 1

        # Get minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the circle around apple
        cv2.circle(image, center, radius, (0, 255, 0), 3)

        # Label each apple with number
        cv2.putText(image, f"{apple_count}", (int(x)-10, int(y)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Display result
cv2.putText(image, f"Total Apples: {apple_count}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

cv2.imshow("Detected Apples", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"Total Apples Detected: {apple_count}")
