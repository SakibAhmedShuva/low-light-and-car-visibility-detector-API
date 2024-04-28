from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import numpy as np
import os
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

prev_light_status = None  # Variable to store previous light status
recording = False
frames = []

model = YOLO("./models/yolov8l.pt")

def check_border_touch(image_width, image_height, cords):
    x_min, y_min, x_max, y_max = cords
    if x_min <= 0 or y_min <= 0 or x_max >= image_width or y_max >= image_height:
        print("The whole car is not visible. Make sure you are not too close to the car and the whole car is being captured.")
        return 'The whole car is not visible. Make sure you are not too close to the car and the whole car is being captured.'
        
    else:
        print("Car is captured properly. Processing...")
        return 'Car is captured properly. Processing...'
 
def calculate_area(image_width, image_height, cords):
    x_min, y_min, x_max, y_max = cords
    car_area = (x_max - x_min) * (y_max - y_min)
    image_area = image_width * image_height
    return car_area / image_area

def check_camera_distance(image_width, image_height, cords):
    car_coverage = calculate_area(image_width, image_height, cords)
    if car_coverage < 1/3:
        print("It seems you are too far from the car. Get close.")
        return 'It seems you are too far from the car. Get close.'
    else:
        print("User distance is okay. Processing...")
        return 'User distance is okay. Processing...'

def check_light(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    avg_brightness = cv2.mean(hsv[:,:,2])[0]
    if avg_brightness < 100:
        return False
    else:
        return True

def save_video():
    global frames
    if len(frames) > 0:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        frames = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    global prev_light_status, recording, frames, model  # Access global variables

    # Get the frame sent from the client
    frame = request.files['frame'].read()

    # Convert frame to numpy array
    nparr = np.frombuffer(frame, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    current_light_status = check_light(frame)

    # Check if the current light status differs from the previous one
    if current_light_status != prev_light_status:
        if current_light_status:
            print('Enough light. Processing video...')
            response = jsonify({'light_status': 'Enough light. Processing video...'})

            # Car Detection
            image = Image.fromarray(frame)
            image = image.resize((640, 320))
            image = np.array(image)
            
            #image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # cv2.imshow('image', image)
            # cv2.waitKey(0)  # Wait until any key is pressed
            # cv2.destroyAllWindows()  # Close the window after any key is pressed
            
            results = model.predict(source=image, imgsz=(640), classes=2,
                        show=True, save=True, show_conf=True, device='cpu')
            try:
                box = results[0].boxes[0]
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                conf = round(box.conf[0].item(), 2)
                print("---")
                print("Coordinates:", cords)
                print("Probability:", conf)
                print("---")

                #height, width, _ = frame.shape
                #car_distance = check_camera_distance(image.width, image.height, cords)
                car_distance = check_camera_distance(image.shape[1], image.shape[0], cords)
                if "far" not in car_distance.lower():  # If the user is not too far from the car, then check border touch
                    border_touch_result = check_border_touch(image.shape[1], image.shape[0], cords)

                    # Return results in JSON format
                    response = jsonify({
                        'light_status': 'Enough light. Processing video...',
                        'car_distance': car_distance,
                        'border_touch_result': border_touch_result
                    })
                else:
                    # Return results indicating that the user is too far from the car
                    response = jsonify({
                        'light_status': 'Enough light. Processing video...',
                        'car_distance': car_distance
                    })

            except IndexError:
                check_result = "Sorry, we cannot detect any car."
                print("Sorry, we cannot detect any car.")
                response = jsonify({'message': check_result})

        else:
            print('Not enough light. Please increase light.')
            response = jsonify({'light_status': 'Not enough light. Please increase light.'})

        # Update the previous light status
        prev_light_status = current_light_status
        return response

    if current_light_status:
        if recording:
            # Resize frame to 720p
            #frame = cv2.resize(frame, (1280, 720))
            frames.append(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            return Response(b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n',
                            mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return jsonify({'light_status': 'Recording stopped.'})
    else:
        return jsonify({'error': 'Not enough light. Recording paused.'}), 500

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    recording = True
    return jsonify({'message': 'Recording started.'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    save_video()
    return jsonify({'message': 'Recording stopped. Video saved.'})

@app.route('/output.mp4')
def download_video():
    video_path = 'output.mp4'  # Path to your video file
    return send_file(video_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)

