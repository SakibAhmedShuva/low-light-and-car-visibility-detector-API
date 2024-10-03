# Low-Light and Car Visibility Detector API

This Flask application provides an API for detecting cars in video streams, assessing lighting conditions, and evaluating car visibility. It uses YOLOv8 for object detection and OpenCV for image processing.

## Features

- Real-time video processing
- Low-light detection
- Car detection using YOLOv8
- Car visibility assessment (distance and border touch)
- Video recording and saving

## Prerequisites

- Python 3.7+
- Flask
- OpenCV (cv2)
- NumPy
- Pillow (PIL)
- Ultralytics YOLOv8

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/SakibAhmedShuva/low-light-and-car-visibility-detector-API.git
   cd low-light-and-car-visibility-detector-API
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the YOLO model:
   - Place the `yolov8l.pt` file in the `models` directory (if not downloaded automatically)

## Usage

1. Start the API server:
   ```
   python app.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`.

3. The application will start processing the video stream from your camera.

4. The API will detect lighting conditions and car visibility in real-time.

## API Endpoints

- `/`: Renders the main page (index.html)
- `/video_feed` (POST): Processes video frames and returns detection results
- `/start_recording` (POST): Starts video recording
- `/stop_recording` (POST): Stops video recording and saves the video
- `/output.mp4`: Downloads the recorded video

## Configuration

You can adjust the following parameters in the code:

- YOLOv8 model path: Change `"./models/yolov8l.pt"` to use a different model
- Image size for processing: Modify `imgsz=(640)` in the `model.predict()` call
- Lighting threshold: Adjust the `avg_brightness < 100` condition in the `check_light()` function

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
