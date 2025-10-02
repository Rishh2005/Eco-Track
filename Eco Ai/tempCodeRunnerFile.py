import os
import speech_recognition as sr
from langdetect import detect
from deep_translator import GoogleTranslator
from groq import Groq
from gtts import gTTS
import playsound

# STEP 1: VOICE TO TEXT
def voice_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Speak...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return None

# STEP 2: LANGUAGE DETECTION
def detect_language(text):
    return detect(text)

# STEP 3: TRANSLATE TO ENGLISH
def translate_to_english(text, lang):
    if lang != "en":
        return GoogleTranslator(source=lang, target="en").translate(text)
    return text

# STEP 4: GROQ LLM INFERENCE
def get_groq_response(prompt):
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"  # or your chosen Groq model
    )
    return response.choices.message.content

# STEP 5: TRANSLATE TO ORIGINAL LANGUAGE
def translate_back(text, lang):
    if lang != "en":
        return GoogleTranslator(source="en", target=lang).translate(text)
    return text

# STEP 6: TEXT TO SPEECH
def speak_text(text, lang):
    tts = gTTS(text=text, lang=lang)
    tts.save("response.mp3")
    playsound.playsound(os.path.abspath("response.mp3"))


# MAIN WORKFLOW
user_text = voice_to_text()
if user_text:
    lang_code = detect_language(user_text)
    english_text = translate_to_english(user_text, lang_code)
    # The prompt can include sustainability context, metals, lifecycle, etc.
    groq_prompt = f"{english_text}\nReply focusing on sustainability and life cycle impacts of metals/mining."
    groq_reply = get_groq_response(groq_prompt)
    reply_in_user_lang = translate_back(groq_reply, lang_code)
    print("AI Assistant:", reply_in_user_lang)
    speak_text(reply_in_user_lang, lang_code)
else:
    print("Could not understand audio.")
