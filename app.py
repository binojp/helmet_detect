from flask import Flask, render_template, Response, request, jsonify, send_from_directory
import cv2
import os
import numpy as np
from ultralytics import YOLO
import time
import threading
from datetime import datetime
import uuid

app = Flask(__name__)

# Configure upload and output directories
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Load the model
try:
    model = YOLO('best_helmet_model.pt')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model or handle error
    model = None

# Video capture object for webcam
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

# Shared variables for real-time processing
latest_frame = None
latest_annotated_frame = None
processing_lock = threading.Lock()

class VideoStream:
    def __init__(self, source=0):
        # CAP_DSHOW for Windows, but fallback to default if it fails
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)
            
        self.stopped = False
        self.new_frame_available = False
        
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
        
        self.proc_thread = threading.Thread(target=self._process, args=())
        self.proc_thread.daemon = True
        self.proc_thread.start()

    def _update(self):
        global latest_frame
        while not self.stopped:
            success, frame = self.cap.read()
            if not success:
                self.stopped = True
                break
            with processing_lock:
                latest_frame = frame
                self.new_frame_available = True
            # Sync with ~30-60 FPS camera
            time.sleep(0.01)

    def _process(self):
        global latest_frame, latest_annotated_frame
        while not self.stopped:
            frame_to_proc = None
            with processing_lock:
                if latest_frame is not None:
                    frame_to_proc = latest_frame.copy()
            
            if frame_to_proc is not None and model:
                # imgsz=160 is a good balance for accuracy vs speed
                results = model(frame_to_proc, conf=0.4, imgsz=160, verbose=False)[0]
                temp_annotated = results.plot()
                with processing_lock:
                    latest_annotated_frame = temp_annotated
            
            # AI doesn't need to run at 100fps. 20-25fps is plenty and saves CPU.
            time.sleep(0.03)

    def read(self):
        with processing_lock:
            self.new_frame_available = False
            # Show AI frame if available, otherwise raw camera
            if latest_annotated_frame is not None:
                return latest_annotated_frame
            return latest_frame

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()

webcam_stream = None

def gen_frames():
    global webcam_stream
    if webcam_stream is None:
        webcam_stream = VideoStream(0)
    
    while not webcam_stream or not webcam_stream.stopped:
        # Only encode and send if there's actually a new frame from the camera
        if not webcam_stream.new_frame_available:
            time.sleep(0.01)
            continue
            
        frame = webcam_stream.read()
        if frame is None:
            continue
            
        # Quality 50 is the sweet spot for balance
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if not ret:
            continue
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Limit to ~30 FPS to avoid overloading the browser network buffer
        time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Could not read image'})
        
        if model:
            results = model(img, conf=0.4, imgsz=640)[0] # Higher res for images
            annotated_img = results.plot()
            
            output_filename = "det_" + filename
            output_path = os.path.join(DETECTION_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_img)
            
            return jsonify({
                'original': filepath.replace('\\', '/'),
                'detected': output_path.replace('\\', '/'),
                'count': len(results.boxes)
            })
        else:
            return jsonify({'error': 'Model not loaded'})

@app.route('/detect_video', methods=['POST'])
def detect_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        return jsonify({
            'filename': filename,
            'status': 'uploaded'
        })
    
    return jsonify({'error': 'Upload failed'})

@app.route('/detect_bulk_images', methods=['POST'])
def detect_bulk_images():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    
    files = request.files.getlist('files')
    results_list = []
    
    for file in files:
        if file.filename == '':
            continue
        
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is not None and model:
            results = model(img, conf=0.4, imgsz=640)[0]
            annotated_img = results.plot()
            
            output_filename = "det_" + filename
            output_path = os.path.join(DETECTION_FOLDER, output_filename)
            cv2.imwrite(output_path, annotated_img)
            
            results_list.append({
                'name': file.filename,
                'detected': output_path.replace('\\', '/'),
                'count': len(results.boxes)
            })
            
    return jsonify({'results': results_list})

@app.route('/detect_bulk_videos', methods=['POST'])
def detect_bulk_videos():
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'})
    
    files = request.files.getlist('files')
    results_list = []
    
    for file in files:
        if file.filename == '':
            continue
        
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        results_list.append({
            'name': file.filename,
            'filename': filename
        })
            
    return jsonify({'results': results_list})

@app.route('/video_stream/<filename>')
def video_stream(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    def gen_video_frames():
        cap = cv2.VideoCapture(filepath)
        # For video files, we can afford slightly better processing
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            # More aggressive frame skipping for slow CPUs
            for _ in range(3): cap.grab() 

            if model:
                results = model(frame, conf=0.35, imgsz=224, verbose=False)[0]
                frame = results.plot()
            
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        cap.release()
        
    return Response(gen_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_webcam')
def stop_webcam():
    global webcam_stream, latest_frame, latest_annotated_frame
    if webcam_stream is not None:
        webcam_stream.stop()
        webcam_stream = None
    latest_frame = None
    latest_annotated_frame = None
    return jsonify({'status': 'stopped'})



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
