import dlib
import numpy np
import cv2
import os
import asyncio
import pyaudio
import tempfile
import json
import edge_tts
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
import pvporcupine
import struct
from vosk import Model, KaldiRecognizer
import serial  # For serial communication
import requests
import time
import base64
import yt_dlp
import webbrowser
import pyautogui
import subprocess
import yagmail
import imaplib
import email
from email.header import decode_header
from bs4 import BeautifulSoup

emotion = ''
api_key_news = '<NEWS_API_KEY>'  # <-- Insert your news API key here
# Load Dlib face detection and recognition models
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(r"<PATH_TO_SHAPE_PREDICTOR>")
face_encoder = dlib.face_recognition_model_v1(r"<PATH_TO_FACE_ENCODER_MODEL>")

def get_latest_email():
    """
    Function to fetch the latest email from the Gmail inbox.
    """
    # Gmail IMAP server details
    email_account = '<EMAIL_ADDRESS>'
    password = '<APP_PASSWORD>'
    imap_server = 'imap.gmail.com'
    
    mail = imaplib.IMAP4_SSL(imap_server)

    try:
        mail.login(email_account, password)
        mail.select("inbox")
        status, messages = mail.search(None, "ALL")
        messages = messages[0].split()
        latest_email_id = messages[-1]
        status, msg_data = mail.fetch(latest_email_id, "(RFC822)")

        subject = None
        sender = None
        body = None

        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")
                sender = msg.get("From")
                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition"))
                        if "attachment" not in content_disposition:
                            if content_type == "text/plain":
                                body = part.get_payload(decode=True).decode()
                else:
                    body = msg.get_payload(decode=True).decode()
        mail.close()
        mail.logout()
        return subject, sender, body

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

def send_email(text, name):
    peoplemail = {
        # "Name": "email@example.com",
        # Add your contacts here
    }
    email_addr = peoplemail.get(name)
    if email_addr:
        yag = yagmail.SMTP("<EMAIL_ADDRESS>", "<APP_PASSWORD>")
        yag.send(to=email_addr, subject="NOVA message", contents=text)
        print("Email sent successfully!")
    else:
        print("No email found for this person.")

def is_chrome_running():
    try:
        result = subprocess.run(
            ["tasklist", "/fi", "imagename eq chrome.exe"], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if "chrome.exe" in result.stdout.decode():
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking for Chrome process: {e}")
        return False

def play_music_from_youtube(query):
    try:
        ydl_opts = {
            'quiet': True,
            'noplaylist': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_results = ydl.extract_info(f"ytsearch:{query}", download=False)
            if 'entries' in search_results and len(search_results['entries']) > 0:
                video = search_results['entries'][0]
                video_url = video['url']
                print(f"Playing: {video['title']}")
                webbrowser.open(video_url)
                time.sleep(10)
                pyautogui.press('space')
                print("Pressed 'Space' to play the video.")
            else:
                print("No results found!")
    except Exception as e:
        print(f"Error: {e}")

def search_google(query):
    api_key = "<GOOGLE_API_KEY>"  # <-- Insert your Google API Key here
    cse_id = "<CUSTOM_SEARCH_ENGINE_ID>"  # <-- Insert your Custom Search Engine ID here
    search_url = "https://www.googleapis.com/customsearch/v1"
    search_params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
    }
    try:
        search_response = requests.get(search_url, params=search_params)
        search_data = search_response.json()
        if "items" not in search_data:
            return "No results found or an error occurred while searching."
        urls = [item["link"] for item in search_data["items"]]
        detailed_info = ""
        for url in urls[:1]:
            try:
                print(f"Extracting content from: {url}")
                page_response = requests.get(url, timeout=10)
                soup = BeautifulSoup(page_response.content, "html.parser")
                paragraphs = soup.find_all("p")
                page_text = "\n".join(p.get_text(strip=True) for p in paragraphs)
                detailed_info += f"\nContent from {url}:\n"
                detailed_info += page_text if page_text else "No relevant text found."
                detailed_info += "\n" + "=" * 80 + "\n"
            except Exception as e:
                detailed_info += f"\nFailed to extract text from {url}: {str(e)}\n"
                detailed_info += "=" * 80 + "\n"
        return detailed_info
    except Exception as e:
        return f"An error occurred while processing the query: {str(e)}"

def encode_faces_from_folder(folder_path):
    known_encodings = []
    known_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detected_faces = detector(img_rgb)
            if len(detected_faces) > 0:
                shape = sp(img_rgb, detected_faces[0])
                face_encoding = face_encoder.compute_face_descriptor(img_rgb, shape)
                known_encodings.append(np.array(face_encoding))
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def get_detailed_weather(location="Singapore"):
    url = f"http://wttr.in/{location}?format=j1"
    try:
        reply = requests.get(url)
        data = reply.json()
        current = data['current_condition'][0]
        current_weather = {
            "temperature": f"{current['temp_C']} °C",
            "feels_like": f"{current['FeelsLikeC']} °C",
            "weather_description": current["weatherDesc"][0]["value"],
            "wind_speed_kph": f"{current['windspeedKmph']} kph",
            "wind_direction": current["winddir16Point"],
            "humidity": f"{current['humidity']}%",
            "pressure": f"{current['pressure']} mb",
            "cloud_cover": f"{current['cloudcover']}%",
            "uv_index": f"{current['uvIndex']}",
            "precipitation_mm": f"{current['precipMM']} mm"
        }
        forecast = []
        for day in data['weather']:
            day_forecast = {
                "date": day["date"],
                "avg_temp_C": f"{day['avgtempC']} °C",
                "avg_temp_F": f"{day['avgtempF']} °F",
                "description": day["hourly"][4]["weatherDesc"][0]["value"],
                "sunrise": day["astronomy"][0]["sunrise"],
                "sunset": day["astronomy"][0]["sunset"],
                "moonrise": day["astronomy"][0].get("moonrise", "N/A"),
                "moonset": day["astronomy"][0].get("moonset", "N/A"),
                "max_temp_C": f"{day['maxtempC']} °C",
                "min_temp_C": f"{day['mintempC']} °C",
            }
            forecast.append(day_forecast)
        return {
            "current_weather": current_weather,
            "forecast": forecast
        }
    except Exception as e:
        print("Error fetching weather data:", e)
        return None

weather_info = get_detailed_weather("Singapore")

class AI_Assistant:
    def __init__(self):
        self.issleep = True
        self.emailcontent = ' '
        self.emailname = ''
        self.emotion = "sleeping"
        self.play_song = False
        self.vision_response = ''
        # Set API keys and access tokens
        self.openai_api_key = "<OPENAI_API_KEY>"
        self.porcupine_access_key = '<PORCUPINE_ACCESS_KEY>'
        self.user_input = ' '
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.porcupine = pvporcupine.create(
            access_key=self.porcupine_access_key,
            keyword_paths=[r'<PATH_TO_HEY_NOVA_PPN>']
        )
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length
        )
        self.transcription_model = Model(r"<PATH_TO_VOSK_MODEL>")
        self.rec_transcribe = KaldiRecognizer(self.transcription_model, 16000)
        self.full_transcript = []
        self.current_memory_file_path = None
        self.is_playing_audio = False
        known_faces_folder = r"<PATH_TO_KNOWN_FACES_FOLDER>"
        self.known_encodings, self.known_names = encode_faces_from_folder(known_faces_folder)
        self.arduino = serial.Serial('<ARDUINO_COM_PORT>', 250000, timeout=1)
        self.transcribing_stream = None

    async def listen_for_trigger(self):
        
        print("Listening for trigger word 'HEY NOVA'...")
        self.arduino.write(b"home,home")
        time.sleep(2)
        self.arduino.write(b"sleeping")
        
        while True:
            pcm = self.stream.read(self.porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
            keyword_index = self.porcupine.process(pcm)
            if keyword_index >= 0:
                self.arduino.write(b"wake")
                self.issleep = False
                # Send "wake" command to Arduino when "Hey Nova" is detected
                
                await self.start_conversation()

    async def start_conversation(self):
        time.sleep(1)
        self.play_mp3('<POWER_ON_SOUND_FILE>')
        await self.align()
        print("aligned")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.6)
        # Capture a single frame from the webcam
        ret, frame = cap.read()
        print("read frame")
        if not ret:
            print("Failed to capture image.")
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        avg_brightness = np.mean(rgb_frame)
        if avg_brightness < 100:
            gamma = 0.5
            rgb_frame = cv2.pow(rgb_frame / 255.0, gamma) * 255.0
            rgb_frame = rgb_frame.astype('uint8')
        else:
            rgb_frame = rgb_frame
        detected_faces = detector(rgb_frame)

        recognized_names = []
        for face in detected_faces:
            shape = sp(rgb_frame, face)
            face_encoding = face_encoder.compute_face_descriptor(rgb_frame, shape)
            face_encoding = np.array(face_encoding)

            distances = np.linalg.norm(self.known_encodings - face_encoding, axis=1)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < 0.6:  # Threshold for recognition
                name = self.known_names[best_match_index]
                recognized_names.append(name)
                self.full_transcript = self.load_memories(name)
            
        cap.release()

        if recognized_names:
            name_list = ', '.join(recognized_names)
            greeting = f"Hello, {name_list}, how can I assist you?"
            self.emailname = name_list
        else:
            greeting = "Hello, how can I assist you?"
            self.full_transcript = self.load_memories("")
        print(greeting)
        await self.generate_audio(greeting)
        await self.start_transcription()

    def load_memories(self, name):
        # Get the current date and time
        now = datetime.now()
        formatted_datetime = now.strftime("%A, %B %d, %Y %I:%M %p")
        
        if name == "":
            self.current_memory_file_path = '<BASIC_MEMORY_FILE_PATH>'
        else:
            
            self.current_memory_file_path = f'<PERSONAL_MEMORY_FILE_PATH>/{name}.txt'
        memories = []
        if os.path.exists(self.current_memory_file_path):
            with open(self.current_memory_file_path, 'r', encoding='utf-8') as file:
                # First, append all the lines to memories
                for line in file:
                    line = line.strip()  # Strip leading/trailing spaces
                    memories.append({
                        "role": "system", 
                        "content": f"Extra info to be used when needed: {line}"
                    })
        
                # After all lines are appended, now append the datetime and weather info
                memories.append({
                    "role": "system", 
                    "content": f"""The date and time today is {formatted_datetime}. The Singapore weather for today and the next 3 days is {weather_info}. Always give concise answers."""
                })
        subject, sender, body = get_latest_email()
        user_input = f"Email sent to you: {body}, from {sender}."
        self.full_transcript.append({"role": "user", "content": user_input})
        return memories


    async def align(self):
        command_x, command_y = '', ''
        last_command = None
        is_search = False

        # Initialize webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Smaller resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        # Initialize dlib's face detector
        detector = dlib.get_frontal_face_detector()

        # Define central region and tolerance
        frame_center_x = 640 // 2
        frame_center_y = 360 // 2
        tolerance = 100  # Smaller tolerance for quicker stopping

        time.sleep(0.3)  # Shorter initialization delay

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting...")
                break

            # Convert frame to grayscale for detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness < 100:
                gamma = 0.5
                gray = cv2.pow(gray / 255.0, gamma) * 255.0
                gray = gray.astype('uint8')
            else:
                gray = gray
            faces = detector(gray)

            if len(faces) == 0:
                if not is_search:
                    self.arduino.write(b"search,search\n")
                    print("Searching for face...")
                    is_search = True
            else:
                is_search = False  # Reset search state
                face = faces[0]  # Process the first detected face

                # Calculate face center
                face_center_x = (face.left() + face.right()) // 2
                face_center_y = (face.top() + face.bottom()) // 2

                # Determine movement commands
                command_x = "right" if face_center_x < frame_center_x - tolerance else \
                            "left" if face_center_x > frame_center_x + tolerance else "stop"

                command_y = "up" if face_center_y < frame_center_y - tolerance else \
                            "down" if face_center_y > frame_center_y + tolerance else "stop"

                # Combine commands and only send if changed
                command = f"{command_x},{command_y}"
                if command != last_command:
                    self.arduino.write(f"{command}\n".encode())
                    print(f"Command sent: {command}")
                    last_command = command

                

            # Stop alignment if centered
            if command_x == "stop" and command_y == "stop":
                print("Alignment complete.")
                break

            # Skip display for speed (or add optional imshow here for debugging)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User terminated alignment.")
                break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("Alignment process finished.")



    async def vision(self, query):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("Failed to open the camera.")
            return
        time.sleep(0.65)
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            return

        # Define the image path
        image_path = "captured_visionimage.jpg"

        # Save the image
        cv2.imwrite(image_path, frame)

        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')

        # Getting the base64 string
        base64_image = encode_image(image_path)

        response = self.openai_client.chat.completions.create(model="gpt-4o-mini",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f'dont talk about the people in the image. {query}',
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}",
                    "detail": "low"
                    
                },
                },
            ],
            }
        ],
        )
        try:
            # Delete the image
            if os.path.exists(image_path):
                os.remove(image_path)
                
            else:
                print("The image file does not exist.")
        except Exception as e:
            print(f"Error deleting the image: {e}")
        
        self.vision_response = response.choices[0].message.content
        print(self.vision_response)



    async def start_transcription(self):
        
        
            


        if self.emotion.lower() == 'inbox':
            subject, sender, body = get_latest_email()
            user_input = f"Email sent to you: {body}, from {sender}. {self.query}"
            self.full_transcript.append({"role": "user", "content": user_input})
            await self.generate_ai_response(user_input)
        # Ensure the transcription stream is initialized only once
        self.arduino.write(b"idle")
        if self.transcribing_stream is None:
            self.transcribing_stream = pyaudio.PyAudio().open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4096
            )
            self.transcribing_stream.start_stream()

        print("Starting real-time transcription...")

        while True:
            data = self.transcribing_stream.read(4096, exception_on_overflow=False)
            if self.rec_transcribe.AcceptWaveform(data):
                result = json.loads(self.rec_transcribe.Result())
                if "text" in result and result != ' ':
                    await self.handle_transcript(result)

    def stop_transcription(self):
        if self.transcribing_stream:
            self.transcribing_stream.stop_stream()
            self.transcribing_stream.close()
            self.transcribing_stream = None

    async def handle_transcript(self, transcript):
        self.stop_transcription()

        user_input = transcript["text"]
        self.user_input = transcript["text"]
        self.full_transcript.append({"role": "user", "content": user_input})
        
        if "play a song" in user_input.lower():
            self.play_song = True
            await self.generate_audio("What song should I play?")
            await self.start_transcription()
            
        if self.play_song:
            play_music_from_youtube(transcript)
            self.play_song = False
        
        if "stop" in user_input.lower() and "song" in user_input.lower():
            if is_chrome_running():
                os.system("taskkill /f /im chrome.exe")  # This will force close Chrome (if on Windows)
            else:
                print("Chrome is not running, cannot close the browser.")

        if 'current' in user_input.lower() or 'now' in user_input.lower():
            results = search_google(user_input)
            user_input = f"Results from web: {results}, Query: {user_input}"
            self.full_transcript.append({"role": "user", "content": user_input})
            
            


        await self.generate_ai_response(user_input)
        await self.start_transcription()

    async def save_memory(self, input_text):
        with open(self.current_memory_file_path, 'a', encoding='utf-8') as file:
            input_text = 'Extra info: ' + input_text
            file.write(f"{input_text}\n")
            

    async def generate_ai_response(self, transcript):
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.full_transcript
        )
        
        ai_response = response.choices[0].message.content
        self.full_transcript.append({"role": "assistant", "content": ai_response})
        # Split the string at the colon
        if ':' in ai_response:
            self.emotion, text_response = ai_response.split(':', 1)  # Split at the first occurrence of ' : '
            
        else:
            # If no colon is present, assume Idle emotion and take the full response as text
            self.emotion = 'Idle'
            text_response = ai_response
            
        print("Response:", text_response)
        self.arduino.write(f"{self.emotion.lower()}".encode())
        if self.emotion.lower() == 'sleeping':
            await self.generate_audio(text_response)
            self.play_mp3('<POWER_OFF_SOUND_FILE>')
            os.system('cls')
            await self.listen_for_trigger()
        if self.emotion.lower() == 'memory':
            await self.save_memory(transcript)
        if self.emotion.lower() == 'vision':
            await self.vision(transcript)
            text_response = self.vision_response
        # Remove personal names, use generic placeholders if needed
        if self.emotion in ['Person1', 'Person2']:
            self.emailname = self.emotion
            self.emailcontent = text_response
            send_email(self.emailcontent, self.emailname)
        
        self.query = transcript  
        self.emailcontent = text_response
        await self.generate_audio(text_response)

    async def generate_audio(self, text):

        self.is_playing_audio = True
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.close()
            communicate = edge_tts.Communicate(text, voice="en-US-GuyNeural")
            await communicate.save(tmp_file.name)
            time.sleep(0.1)
            self.play_mp3(tmp_file.name)
            os.remove(tmp_file.name)
            self.is_playing_audio = False

    def play_mp3(self, path):
        sound = AudioSegment.from_mp3(path)
        play(sound)
        
        

    async def close(self):
        self.stop_transcription()

async def main():
    ai_assistant = AI_Assistant()
    await ai_assistant.generate_audio("NOVA Initialised")
    await ai_assistant.listen_for_trigger()

if __name__ == '__main__':
    asyncio.run(main())
