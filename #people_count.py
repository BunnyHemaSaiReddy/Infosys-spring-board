import cv2
import numpy as np
from ultralytics import YOLO

zone = []
drawing = False
zone_defined = False
use_zone = False

# --- Draw Zone ---
def draw_zone(event, x, y, flags, param):
    global zone, drawing, zone_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        zone = [(x, y)]
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if len(zone) > 1:
            zone[-1] = (x, y)
        else:
            zone.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(zone) > 1:
            zone[-1] = (x, y)
        else:
            zone.append((x, y))
        zone_defined = True


# --- Check if point inside zone ---
def point_in_zone(x, y):
    if len(zone) < 2:
        return False
    x1, y1 = zone[0]
    x2, y2 = zone[-1]
    return min(x1, x2) < x < max(x1, x2) and min(y1, y2) < y < max(y1, y2)


# --- Draw Zone Window ---
def draw_zone_window(frame):
    global zone, zone_defined
    temp = frame.copy()
    cv2.namedWindow("Draw Zone (Drag to create rectangle, press 'c' to confirm)")
    cv2.setMouseCallback("Draw Zone (Drag to create rectangle, press 'c' to confirm)", draw_zone)

    while True:
        temp = frame.copy()
        if len(zone) >= 2:
            cv2.rectangle(temp, zone[0], zone[-1], (255, 0, 0), 2)
        cv2.imshow("Draw Zone (Drag to create rectangle, press 'c' to confirm)", temp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and zone_defined:
            cv2.destroyWindow("Draw Zone (Drag to create rectangle, press 'c' to confirm)")
            break


# --- Process Frame ---
def process_frame(frame, model):
    global use_zone

    results = model(frame, verbose=False)
    total_detected = 0
    inside_zone = 0

    # Draw zone
    if use_zone and len(zone) >= 2:
        cv2.rectangle(frame, zone[0], zone[-1], (255, 0, 0), 2)
        cv2.putText(frame, "ZONE", (zone[0][0], zone[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for result in results:
        boxes = result.boxes.xyxy
        classes = result.boxes.cls
        for i, box in enumerate(boxes):
            cls_id = int(classes[i])
            name = model.names[cls_id]

            # Only consider person and face if available
            if name in ["person", "face"]:
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                total_detected += 1
                color = (0, 255, 0)

                if use_zone and point_in_zone(cx, cy):
                    color = (0, 0, 255)
                    inside_zone += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display counts
    cv2.putText(frame, f"Total Detected: {total_detected}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    if use_zone:
        cv2.putText(frame, f"Inside Zone: {inside_zone}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("YOLO People Detection", frame)


# --- Detection Function ---
def detect_people(source):
    global use_zone

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # lightweight & fast

    choice_zone = input("Do you want to add a zone? (y/n): ").lower()
    use_zone = (choice_zone == 'y')

    # For Image
    if isinstance(source, str) and (source.endswith(".jpg") or source.endswith(".png")):
        image = cv2.imread(source)

        # Resize for fitting screen
        max_width = 800
        if image.shape[1] > max_width:
            scale = max_width / image.shape[1]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if use_zone:
            draw_zone_window(image)

        process_frame(image, model)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # For Video or Webcam
    cap = cv2.VideoCapture(0 if source == 0 else source)
    print("\n🎮 Controls:\nZ - Draw Zone | A - Toggle Zone Counting | Q - Quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for better display
        max_width = 800
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        process_frame(frame, model)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('z'):
            draw_zone_window(frame)
        elif key == ord('a'):
            use_zone = not use_zone
            print(f"🟢 Zone {'Enabled' if use_zone else 'Disabled'}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Main ---
while 1:
    print("Select mode:")
    print("1 - Detect in Image")
    print("2 - Detect in Video")
    print("3 - Use Webcam (press 'q' to quit)")

    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        path = input("Enter image path: ")
        detect_people(path)
    elif choice == '2':
        path = input("Enter video path: ")
        detect_people(path)
    else:
        detect_people(0)