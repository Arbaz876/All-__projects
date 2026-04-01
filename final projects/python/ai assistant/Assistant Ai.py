import sys
import os
import cv2
import numpy as np
import pickle
import time
import traceback
import threading
import requests
import json
import pyttsx3
import speech_recognition as sr
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, 
                             QMessageBox, QComboBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap

# Configuration
CONFIG = {
    'weather_api_key': 'YOUR_OPENWEATHER_API_KEY',
    'news_api_key': 'YOUR_NEWSAPI_KEY',
    'city_name': 'New York',
    'news_country': 'us',
    'wake_word': 'hey jarvis',
    'recognition_threshold': 5,
    'min_confidence': 45,
    'max_confidence': 85,
    'face_data_dir': 'face_data'
}

class FaceRecognizer:
    def __init__(self):
        self.face_cascade = None
        self.recognizer = None
        self.labels = {}
        self.load_models()

    def load_models(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError("Haar cascade file not found")
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            if os.path.exists("face-trainer.yml"):
                self.recognizer.read("face-trainer.yml")
            else:
                raise FileNotFoundError("Training data not found")
                
            if os.path.exists("face-labels.pickle"):
                with open("face-labels.pickle", 'rb') as f:
                    self.labels = pickle.load(f)
                    self.labels = {v:k for k,v in self.labels.items()}
            else:
                raise FileNotFoundError("Label data not found")
                
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def recognize_face(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Improved face detection parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                
                # Resize face to consistent size for better recognition
                roi_gray = cv2.resize(roi_gray, (200, 200), interpolation=cv2.INTER_LINEAR)
                
                # Apply histogram equalization for better contrast
                roi_gray = cv2.equalizeHist(roi_gray)
                
                try:
                    id_, confidence = self.recognizer.predict(roi_gray)
                    if CONFIG['min_confidence'] <= confidence <= CONFIG['max_confidence']:
                        name = self.labels.get(id_, "Unknown")
                        return True, name
                    else:
                        print(f"Confidence too low: {confidence}")
                except Exception as e:
                    print(f"Recognition error: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Face detection error: {str(e)}")
            
        return False, None

class ServiceHandler:
    @staticmethod
    def get_weather(city):
        try:
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}appid={CONFIG['weather_api_key']}&q={city}&units=metric"
            response = requests.get(complete_url)
            data = response.json()
            
            if data["cod"] != "404":
                main = data["main"]
                weather = data["weather"][0]
                temp = main["temp"]
                feels_like = main["feels_like"]
                humidity = main["humidity"]
                description = weather["description"]
                
                return (f"Current weather in {city}: {description}. "
                       f"Temperature is {temp}°C, feels like {feels_like}°C. "
                       f"Humidity at {humidity}%")
            else:
                return "I couldn't find that city. Please try another location."
        except Exception as e:
            print(f"Weather API error: {str(e)}")
            return "Sorry, I'm having trouble accessing weather information right now."

    @staticmethod
    def get_news(country='us', category='general', num_articles=3):
        try:
            url = (f"https://newsapi.org/v2/top-headlines?"
                  f"country={country}&category={category}&"
                  f"apiKey={CONFIG['news_api_key']}")
            response = requests.get(url)
            data = response.json()
            
            if data["status"] == "ok":
                articles = data["articles"][:num_articles]
                news = ["Here are the latest headlines:"]
                for i, article in enumerate(articles, 1):
                    news.append(f"{i}. {article['title']}")
                return " ".join(news)
            else:
                return "I couldn't fetch the news right now."
        except Exception as e:
            print(f"News API error: {str(e)}")
            return "Sorry, I'm having trouble accessing news updates."

    @staticmethod
    def get_time():
        now = datetime.now()
        return f"The time is now {now.strftime('%I:%M %p')}"

    @staticmethod
    def get_date():
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}"

    @staticmethod
    def open_application(app_name):
        apps = {
            'notepad': 'notepad.exe',
            'calculator': 'calc.exe',
            'browser': 'start chrome',
            'paint': 'mspaint.exe',
            'word': 'winword.exe',
            'excel': 'excel.exe',
            'powerpoint': 'powerpnt.exe'
        }
        
        app_name = app_name.lower()
        if app_name in apps:
            try:
                os.system(apps[app_name])
                return f"Opening {app_name} for you now"
            except Exception as e:
                return f"I couldn't open {app_name}"
        else:
            return f"I don't know how to open {app_name}"

class SpeechHandler(QObject):
    speak_signal = pyqtSignal(str)
    wake_detected = pyqtSignal()
    command_received = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.engine = self.init_tts_engine()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False
        self.speak_signal.connect(self.speak)
        
    def init_tts_engine(self):
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        
        # Try to find a natural sounding voice
        preferred_voices = [
            'Microsoft David Desktop',  # Windows
            'Microsoft Zira Desktop',   # Windows
            'english-us',               # Linux
            'english_rp'                # Linux
        ]
        
        for voice in voices:
            if any(v.lower() in voice.name.lower() for v in preferred_voices):
                engine.setProperty('voice', voice.id)
                break
                
        engine.setProperty('rate', 160)
        engine.setProperty('volume', 1.0)
        return engine
        
    def speak(self, text):
        try:
            # Split long responses into chunks for more natural speech
            if len(text) > 120:
                sentences = [s.strip() for s in text.split('. ') if s.strip()]
                for sentence in sentences:
                    self.engine.say(sentence)
                    self.engine.runAndWait()
                    time.sleep(0.15)  # Small pause between sentences
            else:
                self.engine.say(text)
                self.engine.runAndWait()
        except Exception as e:
            print(f"Speech error: {str(e)}")
            
    def listen_for_wake_word(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening for wake word...")
            
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio).lower()
                    
                    if CONFIG['wake_word'] in text:
                        print("Wake word detected!")
                        self.wake_detected.emit()
                        break
                        
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    print("API unavailable")
                    break
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    print(f"Listening error: {str(e)}")
                    break
                    
    def listen_for_command(self):
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Listening for command...")
                self.speak_signal.emit("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=7)
                
            try:
                command = self.recognizer.recognize_google(audio).lower()
                self.command_received.emit(command)
                return command
            except sr.UnknownValueError:
                self.command_received.emit("")
                return ""
            except sr.RequestError:
                self.command_received.emit("")
                return ""
                
        except Exception as e:
            print(f"Command listening error: {str(e)}")
            self.command_received.emit("")
            return ""
            
    def start_listening(self):
        self.listening = True
        threading.Thread(target=self.listen_for_wake_word, daemon=True).start()
        
    def stop_listening(self):
        self.listening = False

class Worker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    update_frame = pyqtSignal(np.ndarray)
    recognition_update = pyqtSignal(str, int)

    def __init__(self, face_recognizer):
        super().__init__()
        self.face_recognizer = face_recognizer
        self.running = True
        self.capture = None

    def run(self):
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Set camera resolution for better face detection
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            while self.running:
                ret, frame = self.capture.read()
                if not ret:
                    self.error.emit("Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                self.update_frame.emit(frame)
                
                if int(time.time() * 10) % 5 == 0:
                    recognized, name = self.face_recognizer.recognize_face(frame)
                    if recognized:
                        self.recognition_update.emit(name, 1)
                    else:
                        self.recognition_update.emit("", -1)
                
                QThread.msleep(30)
                
        except Exception as e:
            self.error.emit(f"Camera error: {str(e)}")
            print(traceback.format_exc())
        finally:
            if self.capture:
                self.capture.release()
            self.finished.emit()

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("JARVIS - Personal Assistant")
        self.setGeometry(100, 100, 1000, 700)
        
        try:
            self.face_recognizer = FaceRecognizer()
        except Exception as e:
            self.show_error_message(f"Failed to initialize face recognition: {str(e)}")
            sys.exit(1)
            
        self.authenticated = False
        self.user_name = ""
        self.recognition_counter = 0
        self.greeted = False
        self.continuous_listening = False
        
        self.init_ui()
        self.init_services()
        self.init_camera_thread()

    def show_error_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(message)
        msg.setWindowTitle("Error")
        msg.exec_()

    def init_ui(self):
        main_widget = QWidget()
        layout = QHBoxLayout()
        
        # Left panel - Camera
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(640, 480)
        left_layout.addWidget(self.camera_label)
        
        self.status_label = QLabel("Initializing camera...")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)
        
        left_panel.setLayout(left_layout)
        
        # Right panel - Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        
        # Voice controls
        self.voice_toggle = QPushButton("Enable Continuous Listening")
        self.voice_toggle.clicked.connect(self.toggle_continuous_listening)
        right_layout.addWidget(self.voice_toggle)
        
        # Settings
        settings_group = QWidget()
        settings_layout = QHBoxLayout()
        
        self.city_combo = QComboBox()
        self.city_combo.addItems(["New York", "London", "Tokyo", "Paris", "Mumbai"])
        self.city_combo.setCurrentText(CONFIG['city_name'])
        settings_layout.addWidget(QLabel("City:"))
        settings_layout.addWidget(self.city_combo)
        
        self.country_combo = QComboBox()
        self.country_combo.addItems(["us", "gb", "in", "fr", "jp"])
        self.country_combo.setCurrentText(CONFIG['news_country'])
        settings_layout.addWidget(QLabel("News Country:"))
        settings_layout.addWidget(self.country_combo)
        
        settings_group.setLayout(settings_layout)
        right_layout.addWidget(settings_group)
        
        # Response area
        self.response_label = QLabel("JARVIS responses will appear here")
        self.response_label.setWordWrap(True)
        self.response_label.setStyleSheet("""
            border: 1px solid #2c3e50;
            padding: 10px;
            border-radius: 5px;
            background-color: #f8f9fa;
            color: #2c3e50;
            font-size: 14px;
        """)
        right_layout.addWidget(self.response_label)
        
        # Debug console
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setStyleSheet("""
            background-color: #2c3e50;
            color: #ecf0f1;
            font-family: Consolas;
            font-size: 12px;
        """)
        right_layout.addWidget(self.debug_text)
        
        right_panel.setLayout(right_layout)
        
        layout.addWidget(left_panel)
        layout.addWidget(right_panel)
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def init_services(self):
        self.speech_handler = SpeechHandler()
        self.speech_handler.wake_detected.connect(self.on_wake_detected)
        self.speech_handler.command_received.connect(self.process_voice_command)
        
    def init_camera_thread(self):
        self.thread = QThread()
        self.worker = Worker(self.face_recognizer)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.handle_error)
        self.worker.update_frame.connect(self.update_frame_display)
        self.worker.recognition_update.connect(self.update_recognition_status)
        
        self.thread.start()

    def update_frame_display(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Improved face detection parameters
            faces = self.face_recognizer.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            for (x, y, w, h) in faces:
                if self.authenticated:
                    color = (0, 255, 0)  # Green
                elif self.recognition_counter > 0:
                    color = (0, 165, 255)  # Orange
                else:
                    color = (0, 0, 255)  # Red
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Add text label
                label = "Authenticated" if self.authenticated else "Recognizing..." if self.recognition_counter > 0 else "Unknown"
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(q_img))
            
        except Exception as e:
            self.debug_text.append(f"Display error: {str(e)}")

    def update_recognition_status(self, name, increment):
        if not self.authenticated:
            if name:
                self.user_name = name
                self.recognition_counter += increment
                self.status_label.setText(
                    f"Recognized: {name} ({self.recognition_counter}/{CONFIG['recognition_threshold']})"
                )
                
                if self.recognition_counter >= CONFIG['recognition_threshold']:
                    self.authenticated = True
                    self.status_label.setText(f"Authenticated: {name}")
                    self.debug_text.append(f"User {name} authenticated successfully")
                    greeting = (f"Welcome back {name}! I am JARVIS, your personal assistant. "
                               "You can say 'Hey JARVIS' followed by your command, "
                               "or ask me for help to see what I can do.")
                    self.update_response_display("", greeting)
                    self.speech_handler.speak_signal.emit(greeting)
                    self.greeted = True
                    if self.continuous_listening:
                        self.speech_handler.start_listening()
            else:
                self.recognition_counter = max(0, self.recognition_counter + increment)
                if self.recognition_counter > 0:
                    self.status_label.setText(
                        f"Recognizing... ({self.recognition_counter}/{CONFIG['recognition_threshold']})"
                    )
                else:
                    self.status_label.setText("Please look at the camera")

    def toggle_continuous_listening(self):
        self.continuous_listening = not self.continuous_listening
        
        if self.continuous_listening:
            if not self.authenticated:
                response = "Please authenticate first before enabling continuous listening"
                self.update_response_display("", response)
                self.speech_handler.speak_signal.emit(response)
                self.continuous_listening = False
                self.voice_toggle.setText("Enable Continuous Listening")
                return
                
            self.voice_toggle.setText("Disable Continuous Listening")
            self.speech_handler.start_listening()
            response = f"I'm now listening for the wake phrase '{CONFIG['wake_word']}'"
            self.update_response_display("", response)
            self.speech_handler.speak_signal.emit(response)
        else:
            self.voice_toggle.setText("Enable Continuous Listening")
            self.speech_handler.stop_listening()
            response = "Continuous listening disabled"
            self.update_response_display("", response)
            self.speech_handler.speak_signal.emit(response)

    def on_wake_detected(self):
        response = "How may I assist you?"
        self.update_response_display("", response)
        self.speech_handler.speak_signal.emit(response)
        threading.Thread(target=self.speech_handler.listen_for_command, daemon=True).start()

    def process_voice_command(self, command):
        if not command:
            response = "I didn't catch that. Could you please repeat your request?"
            self.update_response_display("", response)
            self.speech_handler.speak_signal.emit(response)
            return
            
        response = self.generate_response(command)
        self.update_response_display(command, response)
        self.speech_handler.speak_signal.emit(response)
        
        if self.continuous_listening:
            self.speech_handler.start_listening()

    def generate_response(self, command):
        command = command.lower()
        
        if not command:
            return "I didn't hear anything. Could you please repeat that?"
        
        # Basic commands
        if any(greet in command for greet in ['hello', 'hi', 'hey']):
            return f"Hello {self.user_name}! How can I be of service today?"
            
        elif "how are you" in command:
            return "I'm functioning perfectly, thank you for asking! How may I assist you?"
            
        elif "thank" in command:
            return "You're most welcome! Is there anything else I can help with?"
            
        elif any(bye in command for bye in ['bye', 'goodbye']):
            self.greeted = False
            return f"Goodbye {self.user_name}! Have a wonderful day."
            
        elif "your name" in command:
            return "I am JARVIS, your personal assistant. Always at your service!"
            
        # Time and date
        elif "time" in command:
            return ServiceHandler.get_time()
            
        elif "date" in command:
            return ServiceHandler.get_date()
            
        # Weather
        elif "weather" in command:
            city = self.extract_entity(command, ['in', 'for', 'at'])
            city = city if city else self.city_combo.currentText()
            return ServiceHandler.get_weather(city)
            
        # News
        elif "news" in command:
            country = self.extract_entity(command, ['from', 'in', 'about'])
            country = country if country else self.country_combo.currentText()
            return ServiceHandler.get_news(country)
            
        # Applications
        elif "open" in command:
            app = command.replace("open", "").strip()
            return ServiceHandler.open_application(app)
            
        # Settings
        elif "change city" in command or "set city" in command:
            city = self.extract_entity(command, ['to', 'as'])
            if city:
                self.city_combo.setCurrentText(city.capitalize())
                return f"I've updated your preferred city to {city}"
            else:
                return "Please specify which city you'd like me to set"
                
        elif "change country" in command or "set country" in command:
            country = self.extract_entity(command, ['to', 'as'])
            if country:
                self.country_combo.setCurrentText(country.lower())
                return f"I've updated your news country to {country}"
            else:
                return "Please specify which country's news you'd prefer"
                
        # System control
        elif any(cmd in command for cmd in ['stop listening', 'disable listening']):
            self.continuous_listening = False
            self.voice_toggle.setText("Enable Continuous Listening")
            self.speech_handler.stop_listening()
            return "I've disabled continuous listening mode"
            
        elif any(cmd in command for cmd in ['start listening', 'enable listening']):
            self.continuous_listening = True
            self.voice_toggle.setText("Disable Continuous Listening")
            self.speech_handler.start_listening()
            return "I've enabled continuous listening mode"
            
        # Help command
        elif "help" in command or "what can you do" in command:
            return ("I can assist you with various tasks:\n"
                   "• Tell you the current time and date\n"
                   "• Provide weather forecasts for any city\n"
                   "• Give you the latest news headlines\n"
                   "• Open applications like Notepad or Calculator\n"
                   "• Change settings like your preferred location\n"
                   "Just ask and I'll do my best to help!")
            
        else:
            return "I'm not quite sure I understand. Could you please rephrase your request?"

    def update_response_display(self, command, response):
        if command:
            display_text = f"<b>You:</b> {command}<br><br><b>JARVIS:</b> {response}"
        else:
            display_text = f"<b>JARVIS:</b> {response}"
        self.response_label.setText(display_text)

    def extract_entity(self, command, keywords):
        for keyword in keywords:
            if keyword in command:
                parts = command.split(keyword)
                if len(parts) > 1:
                    return parts[1].strip()
        return None

    def handle_error(self, message):
        self.status_label.setText("Error occurred")
        self.debug_text.append(f"ERROR: {message}")
        QMessageBox.critical(self, "Error", message)

    def closeEvent(self, event):
        if hasattr(self, 'worker'):
            self.worker.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        if hasattr(self, 'speech_handler'):
            self.speech_handler.stop_listening()
            self.speech_handler.engine.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    def excepthook(exctype, value, traceback):
        error_msg = ''.join(traceback.format_exception(exctype, value, traceback))
        print(error_msg)
        QMessageBox.critical(None, "Error", str(value))
    
    sys.excepthook = excepthook
    
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)
