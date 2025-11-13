import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math

# --- Setup ---
if not os.path.exists("saved_drawings"):
    os.makedirs("saved_drawings")

# Mediapipe Hand Tracking (with segmentation)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
)
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Canvas
canvas = None
prev_x, prev_y = 0, 0

# Brush & Eraser
brush_color = (0, 0, 255)  # Red by default
brush_thickness = 15
eraser_thickness = 100
active_tool = "brush"

# Color Palette
palette = [
    (20, 20, 70, 70, (0, 0, 255), "Red"),
    (80, 20, 130, 70, (0, 255, 0), "Green"),
    (140, 20, 190, 70, (255, 0, 0), "Blue"),
    (200, 20, 250, 70, (0, 255, 255), "Yellow"),
]
selected_color = brush_color

# Tool Buttons
tools = [
    (300, 20, 350, 70, "brush", "Brush"),
    (360, 20, 410, 70, "eraser", "Eraser"),
    (460, 20, 510, 70, "clear", "Clear"),
    (520, 20, 570, 70, "save", "Save"),
]

# Click control
CLICK_COOLDOWN = 0.5
last_click_time = 0

# Message display
message = ""
message_time = 0
MESSAGE_DURATION = 2.0


def show_message(text):
    global message, message_time
    message = text
    message_time = time.time()


# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw UI Bar
    ui_bar = frame[0:90, 0:w]
    white_rect = np.ones(ui_bar.shape, dtype=np.uint8) * 255
    res = cv2.addWeighted(ui_bar, 0.5, white_rect, 0.5, 1.0)
    frame[0:90, 0:w] = res

    # Draw Palette
    for x1, y1, x2, y2, color, name in palette:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        if selected_color == color:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 3)

    # Draw Tools
    for x1, y1, x2, y2, tool_name, name in tools:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), -1)
        cv2.putText(frame, name, (x1 + 5, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if active_tool == tool_name:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 0, 0), 3)

    # Convert frame to RGB and process with MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    segmentation_mask = getattr(results, 'segmentation_mask', None)

    # ... (inside the main loop, after results = hands.process(rgb)) ...

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Get coordinates for index and middle fingertips
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Convert normalized (0.0-1.0) coords to pixel coords
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            mx, my = int(middle_tip.x * w), int(middle_tip.y * h)

            # --- Gesture Detection ---
            # 1. Find distance between index and middle finger
            distance = math.hypot(mx - ix, my - iy)

            # 2. Check if fingers are "up" (a simple check: tip is above the MCP joint)
            index_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * h
            middle_mcp_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h

            # Gesture 1: Selection Mode (Index and Middle are up and close)
            # We use 'ix, iy' as the cursor position
            if (iy < index_mcp_y) and (my < middle_mcp_y) and (distance < 40):
                # This is "Selection Mode"
                cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)  # Show cursor as "selecting"

                # Check for UI clicks (only if in the UI bar)
                if iy < 90 and (time.time() - last_click_time) > CLICK_COOLDOWN:
                    last_click_time = time.time()

                    # Check Palette
                    for x1, y1, x2, y2, color, _ in palette:
                        if x1 < ix < x2:
                            selected_color = color
                            brush_color = selected_color
                            show_message("Color Changed")
                            break

                    # Check Tools
                    for x1, y1, x2, y2, tool_name, name in tools:
                        if x1 < ix < x2:
                            if tool_name == "clear":
                                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                                show_message("Canvas Cleared")
                            elif tool_name == "save":
                                filename = f"saved_drawings/drawing_{int(time.time())}.png"
                                cv2.imwrite(filename, canvas)
                                show_message("Saved Drawing!")
                            else:
                                active_tool = tool_name
                                show_message(f"{name} Selected")
                            break

                # Reset prev_x, prev_y so we don't draw a line when we switch back to drawing
                prev_x, prev_y = 0, 0

            # Gesture 2: Drawing Mode (Only Index is up)
            elif (iy < index_mcp_y) and (my > middle_mcp_y):
                # This is "Drawing Mode"
                cv2.circle(frame, (ix, iy), 10, (255, 0, 255), -1)  # Show cursor as "drawing"

                # If we just started drawing, set the 'prev' points to the current point
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = ix, iy

                # Draw a line from the previous point to the current point
                if active_tool == "brush":
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), brush_color, brush_thickness)
                elif active_tool == "eraser":
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), (0, 0, 0), eraser_thickness)

                # Update the previous point
                prev_x, prev_y = ix, iy

            # Gesture 3: Moving (No gesture detected)
            else:
                # Reset prev_x, prev_y so we don't draw a line from the last known point
                prev_x, prev_y = 0, 0

    else:
        # No hand detected, reset drawing points
        prev_x, prev_y = 0, 0

    # --- (The rest of your code from "Merge canvas..." is fine) ---

    # Merge canvas with webcam frame
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask_inv = cv2.threshold(canvas_gray, 0, 255, cv2.THRESH_BINARY_INV)
    frame_part = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_part = cv2.bitwise_and(canvas, canvas, mask=cv2.bitwise_not(mask_inv))
    output = cv2.add(frame_part, canvas_part)

    # Display message
    if message and (time.time() - message_time) < MESSAGE_DURATION:
        cv2.putText(output, message, (w // 2 - 100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        message = ""

    cv2.putText(output, "Mode: Segmentation", (10, h - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("AI-Based Virtual Painter (Segmentation)", output)

    # Optional: show hand mask for verification
    if segmentation_mask is not None:
        cv2.imshow("Hand Segmentation", (segmentation_mask * 255).astype(np.uint8))

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
