import speech_recognition as sr

def audio_to_text(audio_file):
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Load the audio file
    with sr.AudioFile(audio_file) as source:
        # Adjust for ambient noise
        recognizer.adjust_for_ambient_noise(source)
        
        # Record the audio
        audio_data = recognizer.record(source)
        
        # Convert speech to text
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))

# Example usage
audio_file = "C:/Users/sriva/NLP/20 Second Timer with Voice Countdown.wav"
text = audio_to_text(audio_file)
print("Transcribed Text:", text)
