import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("--- Available Voices on Your Mac ---")
for index, voice in enumerate(voices):
    print(f"Voice {index}:")
    print(f"  Name: {voice.name}")
    print(f"  ID:   {voice.id}\n")