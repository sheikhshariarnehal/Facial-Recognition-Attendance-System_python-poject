import cv2
import os
from flask import Flask, request, render_template, jsonify
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttendanceConfig:
    """Configuration class for attendance system"""
    REQUIRED_IMAGES: int = 10
    MIN_FACE_SIZE: Tuple[int, int] = (20, 20)
    MODEL_FILENAME: str = 'face_recognition_model.pkl'
    CASCADE_FILE: str = 'haarcascade_frontalface_default.xml'
    IMAGE_SIZE: Tuple[int, int] = (50, 50)
    
class AttendanceSystem:
    def __init__(self, config: AttendanceConfig):
        self.config = config
        self.app = Flask(__name__)
        self.setup_directories()
        self.face_detector = cv2.CascadeClassifier(self.config.CASCADE_FILE)
        self.background = cv2.imread("background.png")
        self.setup_routes()
        
    def setup_directories(self):
        """Create necessary directories if they don't exist"""
        directories = ['Attendance', 'static', 'static/faces']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            
        # Initialize attendance file if it doesn't exist
        self.datetoday = date.today().strftime("%m_%d_%y")
        attendance_file = Path(f'Attendance/Attendance-{self.datetoday}.csv')
        if not attendance_file.exists():
            attendance_file.write_text('Name,Roll,Time\n')
            
    def extract_faces(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Extract faces from image using cascade classifier"""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5, 
                minSize=self.config.MIN_FACE_SIZE
            )
        except Exception as e:
            logger.error(f"Error extracting faces: {e}")
            return []
            
    def identify_face(self, face_array: np.ndarray) -> str:
        """Identify face using trained model"""
        try:
            model = joblib.load(f'static/{self.config.MODEL_FILENAME}')
            return model.predict(face_array)[0]
        except Exception as e:
            logger.error(f"Error identifying face: {e}")
            return "Unknown"
            
    def train_model(self):
        """Train the face recognition model"""
        try:
            faces = []
            labels = []
            faces_dir = Path('static/faces')
            
            for user_dir in faces_dir.iterdir():
                for img_path in user_dir.glob('*.jpg'):
                    img = cv2.imread(str(img_path))
                    resized_face = cv2.resize(img, self.config.IMAGE_SIZE)
                    faces.append(resized_face.ravel())
                    labels.append(user_dir.name)
                    
            if faces and labels:
                faces = np.array(faces)
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(faces, labels)
                joblib.dump(knn, f'static/{self.config.MODEL_FILENAME}')
                logger.info("Model trained successfully")
            else:
                logger.warning("No faces found for training")
                
        except Exception as e:
            logger.error(f"Error training model: {e}")
            
    def add_attendance(self, name: str):
        """Add attendance record"""
        try:
            username, userid = name.split('_')
            current_time = datetime.now().strftime("%H:%M:%S")
            
            df = pd.read_csv(f'Attendance/Attendance-{self.datetoday}.csv')
            if int(userid) not in df['Roll'].values:
                with open(f'Attendance/Attendance-{self.datetoday}.csv', 'a') as f:
                    f.write(f'\n{username},{userid},{current_time}')
                logger.info(f"Attendance marked for {username}")
        except Exception as e:
            logger.error(f"Error adding attendance: {e}")
            
    def setup_routes(self):
        @self.app.route('/')
        def home():
            try:
                df = pd.read_csv(f'Attendance/Attendance-{self.datetoday}.csv')
                datetoday2 = date.today().strftime("%d-%B-%Y")
                return render_template(
                    'home.html',
                    names=df['Name'].tolist(),
                    rolls=df['Roll'].tolist(),
                    times=df['Time'].tolist(),
                    l=len(df),
                    totalreg=len(list(Path('static/faces').iterdir())),
                    datetoday2=datetoday2
                )
            except Exception as e:
                logger.error(f"Error rendering home page: {e}")
                return "Error loading page", 500

        @self.app.route('/start')
        def start():
            try:
                if not Path(f'static/{self.config.MODEL_FILENAME}').exists():
                    return jsonify({"error": "No trained model found"})
                    
                cap = cv2.VideoCapture(0)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    faces = self.extract_faces(frame)
                    for (x, y, w, h) in faces:
                        face = cv2.resize(frame[y:y+h, x:x+w], self.config.IMAGE_SIZE)
                        identified_person = self.identify_face(face.reshape(1, -1))
                        if identified_person != "Unknown":
                            self.add_attendance(identified_person)
                            
                        # Draw face rectangle and name
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (50,50,255), 2)
                        cv2.putText(frame, identified_person, (x,y-10), 
                                  cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1)
                        
                    # Display frame
                    self.background[162:162+480, 55:55+640] = frame
                    cv2.imshow('Attendance', self.background)
                    
                    if cv2.waitKey(1) == 27:  # ESC key
                        break
                        
                cap.release()
                cv2.destroyAllWindows()
                return self.app.redirect('/')
                
            except Exception as e:
                logger.error(f"Error in start route: {e}")
                return jsonify({"error": str(e)})

        @self.app.route('/add', methods=['POST'])
        def add():
            try:
                username = request.form['newusername']
                userid = request.form['newuserid']
                user_dir = Path(f'static/faces/{username}_{userid}')
                user_dir.mkdir(exist_ok=True)
                
                cap = cv2.VideoCapture(0)
                image_count = 0
                frame_count = 0
                
                while image_count < self.config.REQUIRED_IMAGES:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    faces = self.extract_faces(frame)
                    for (x, y, w, h) in faces:
                        if frame_count % 5 == 0:
                            face_img = frame[y:y+h, x:x+w]
                            img_path = user_dir / f'{username}_{image_count}.jpg'
                            cv2.imwrite(str(img_path), face_img)
                            image_count += 1
                            
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,20), 2)
                        cv2.putText(frame, f'Images Captured: {image_count}/{self.config.REQUIRED_IMAGES}',
                                  (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,20), 2)
                        
                    frame_count += 1
                    cv2.imshow('Adding New User', frame)
                    
                    if cv2.waitKey(1) == 27:
                        break
                        
                cap.release()
                cv2.destroyAllWindows()
                
                self.train_model()
                return self.app.redirect('/')
                
            except Exception as e:
                logger.error(f"Error adding new user: {e}")
                return jsonify({"error": str(e)})

    def run(self, debug=True):
        self.app.run(debug=debug)

if __name__ == "__main__":
    config = AttendanceConfig()
    system = AttendanceSystem(config)
    system.run()
