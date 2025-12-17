#voice controlled vision
from ultralytics import YOLO    
from openai import OpenAI
import cv2
import base64
import tempfile
import pyttsx3
import speech_recognition as sr
import time

model = YOLO('yolov8s.pt')
client = OpenAI(api_key="your-api-key")

def ask_vlm(frame, question):
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        cv2.imwrite(f.name, frame)
        temp_path = f.name

    with open(temp_path,'rb') as f:
        image = base64.b64encode(f.read()).decode('utf-8')
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{image}"
                }}
            ]
        }]
    )
    return response.choices[0].message.content

cap = cv2.VideoCapture(1)
prev_time = time.time()
recognizer = sr.Recognizer()

print("System ready. Say 'analyze' to ask VLM about the scene.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # YOLO detection
    results = model(frame)
    annotated = results[0].plot()
    
    # Add text
    cv2.putText(annotated, f'FPS: {int(fps)}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, 'Say "analyze" to ask VLM', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, 'Press Q to quit', (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Voice-Controlled Vision', annotated)
    
    # Listen for voice command
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.1)
        try:
            audio = recognizer.listen(source, timeout=0.5, phrase_time_limit=2)
            command = recognizer.recognize_whisper(audio).lower()
            
            if 'analyze' in command or 'what do you see' in command:
                print("Analyzing scene...")
                description = ask_vlm(frame, 'Describe the objects in the scene.')
                print(f"\nVLM: {description}\n")
                
                #init tts engine
                engine = pyttsx3.init()
                engine.say(description)
                engine.runAndWait()
                
        except:
            pass
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()