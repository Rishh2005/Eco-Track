import os
from dotenv import load_dotenv
from groq import Groq
import speech_recognition as sr
from gtts import gTTS
import pygame
from langdetect import detect
from deep_translator import GoogleTranslator

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
indian_langs = ['hi', 'ta', 'bn', 'mr', 'gu', 'kn', 'ml', 'te', 'ur']

def speak(text, lang='en'):
    if lang not in indian_langs and lang != 'en':
        lang = 'en'
    print(f"Assistant: {text}")
    tts = gTTS(text=text, lang=lang)
    filename = "response.mp3"
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    os.remove(filename)

def get_audio(language="en-IN"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        print(f"Listening ({language}) ...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language=language)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand.")
            return None
        except sr.RequestError as e:
            print(f"Speech service error: {e}")
            return None

def detect_lang(text):
    try:
        return detect(text)
    except Exception:
        return 'en'

def translate(text, src, dest):
    if src == dest:
        return text
    try:
        return GoogleTranslator(source=src, target=dest).translate(text)
    except Exception:
        return text

def main():
    pygame.init()
    while True:
        mode = input("Type 't' (text), 'v' (voice), or 'q' (quit): ").strip().lower()
        if mode == 'q':
            break
        user_text = None
        if mode == 't':
            user_text = input("Enter your message: ")
        elif mode == 'v':
            lang_choice = input("Language? (hi, ta, bn, mr, gu, kn, ml, te, ur, en): ").strip().lower()
            sr_lang = f"{lang_choice}-IN" if lang_choice in indian_langs or lang_choice == "en" else "en-IN"
            user_text = get_audio(language=sr_lang)
            if not user_text:
                continue
        else:
            print("Invalid input.")
            continue

        detected_lang = detect_lang(user_text)
        print(f"Detected language: {detected_lang}")
        english_text = translate(user_text, detected_lang, 'en')
        prompt = f"{english_text}\nReply as a thoughtful sustainability assistant for metals/mines."

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"
        )
        reply_text_en = response.choices[0].message.content
        reply_text_user_lang = translate(reply_text_en, 'en', detected_lang)
        print(f"\nAssistant ({detected_lang}): {reply_text_user_lang}")
        speak(reply_text_user_lang, lang=detected_lang)
    pygame.quit()

if __name__ == "__main__":
    main()
