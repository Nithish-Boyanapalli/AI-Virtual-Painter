# AI-Based Virtual Painter

A virtual drawing application that uses your hand as a brush, built with Python, OpenCV, and MediaPipe.

## Features

* **Hand Tracking:** Uses MediaPipe to detect hand landmarks.
* **Gesture Control:**
    * **Drawing Mode:** Raise your index finger to draw.
    * **Selection Mode:** Raise your index and middle fingers (pinching gesture) to select tools.
* **Tools & Colors:**
    * Multi-color palette (Red, Green, Blue, Yellow)
    * Brush tool
    * Eraser tool
* **Canvas Controls:**
    * Clear the entire canvas
    * Save the current drawing as a `.png` file

## How to Run

1.  Clone this repository.
2.  Install the required libraries:
    ```bash
    pip install opencv-python mediapipe numpy
    ```
3.  Run the main script:
    ```bash
    python virtual_painter.py
    ```
4.  Press 'Esc' to quit.
