
import streamlit as st
from pytube import YouTube
from pytube.exceptions import RegexMatchError
import os
import whisper
import nltk
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
from googletrans import Translator

def write_text_to_file(text, output_file):
    with open(output_file, 'w') as file:
        file.write(text)

def download_audio_stream(url):
    try:
        yt = YouTube(url)
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        if audio_stream:
            return audio_stream
        else:
            st.warning("No suitable audio stream found.")
            return None
    except RegexMatchError:
        st.error("Invalid YouTube URL.")
        return None

def download_and_convert_audio(url, output_path='downloads'):
    # Download the audio stream
    audio_stream = download_audio_stream(url)
    if audio_stream:
        # Set the output path for the WAV file
        wav_output_path = os.path.join(output_path, audio_stream.title + '.wav')
        # Download the audio in WAV format
        audio_stream.download(output_path=output_path, filename=audio_stream.title + '.mp4')

        # Rename the downloaded MP4 file to WAV
        os.rename(os.path.join(output_path, audio_stream.title + '.mp4'), wav_output_path)

        # Print a message indicating the conversion is complete
        st.success(f"Audio downloaded and saved as WAV. Saved at {wav_output_path}")
        return wav_output_path;

def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(file_name, top_n):
    nltk.download("stopwords")
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and split it
    sentences =  read_article(file_name)

    # Step 2 - Generate Similarity Matrix across sentences
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity matrix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    

    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Output the summarized text
    summarized_text = ". ".join(summarize_text)
    return summarized_text

def translate_file_to_language(file_path, target_language):
    translator = Translator()
    
    # Read the content of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Translate the text
    translation = translator.translate(text, dest=target_language)
    
    # Return the translated text
    return translation.text

# Streamlit app
def main():
    st.title("YouTube Audio Summarizer and Translator")

    # Input field for YouTube URL
    youtube_url = st.text_input("Paste YouTube URL:")
    if youtube_url:
        if st.button("Generate Summary"):
            # Download and convert audio
            audio_path = download_and_convert_audio(youtube_url)

            # Transcribe audio
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, fp16=False)
            transcribed_text = result['text']

            # Write transcribed text to file
            transcribed_file_path = "transcribed_text.txt"
            write_text_to_file(transcribed_text, transcribed_file_path)

            # Generate summary
            summary = generate_summary(transcribed_file_path, 5)

            # Write summarized text to file
            summarized_file_path = "summarized_text.txt"
            write_text_to_file(summary, summarized_file_path)
            st.write(summary)

            # Language selection for translation
        x = "summarized_text.txt"
        st.header("Select target language for translation:")
        languages = {
            "af": "Afrikaans",
            "sq": "Albanian",
            "am": "Amharic",
            "ar": "Arabic",
            "hy": "Armenian",
            "az": "Azerbaijani",
            "eu": "Basque",
            "be": "Belarusian",
            "bn": "Bengali",
            "bs": "Bosnian",
            "bg": "Bulgarian",
            "ca": "Catalan",
            "ceb": "Cebuano",
            "ny": "Chichewa",
            "zh-cn": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "co": "Corsican",
            "hr": "Croatian",
            "cs": "Czech",
            "da": "Danish",
            "nl": "Dutch",
            "en": "English",
            "eo": "Esperanto",
            "et": "Estonian",
            "tl": "Filipino",
            "fi": "Finnish",
            "fr": "French",
            "fy": "Frisian",
            "gl": "Galician",
            "ka": "Georgian",
            "de": "German",
            "el": "Greek",
            "gu": "Gujarati",
            "ht": "Haitian Creole",
            "ha": "Hausa",
            "haw": "Hawaiian",
            "iw": "Hebrew",
            "he": "Hebrew",
            "hi": "Hindi",
            "hmn": "Hmong",
            "hu": "Hungarian",
            "is": "Icelandic",
            "ig": "Igbo",
            "id": "Indonesian",
            "ga": "Irish",
            "it": "Italian",
            "ja": "Japanese",
            "jw": "Javanese",
            "kn": "Kannada",
            "kk": "Kazakh",
            "km": "Khmer",
            "ko": "Korean",
            "ku": "Kurdish (Kurmanji)",
            "ky": "Kyrgyz",
            "lo": "Lao",
            "la": "Latin",
            "lv": "Latvian",
            "lt": "Lithuanian",
            "lb": "Luxembourgish",
            "mk": "Macedonian",
            "mg": "Malagasy",
            "ms": "Malay",
            "ml": "Malayalam",
            "mt": "Maltese",
            "mi": "Maori",
            "mr": "Marathi",
            "mn": "Mongolian",
            "my": "Myanmar (Burmese)",
            "ne": "Nepali",
            "no": "Norwegian",
            "or": "Odia",
            "ps": "Pashto",
            "fa": "Persian",
            "pl": "Polish",
            "pt": "Portuguese",
            "pa": "Punjabi",
            "ro": "Romani",
            "ru": "Russian",
            "sm": "Samoan",
            "gd": "Scots Gaelic",
            "sr": "Serbian",
            "st": "Sesotho",
            "sn": "Shona",
            "sd": "Sindhi",
            "si": "Sinhala",
            "sk": "Slovak",
            "sl": "Slovenian",
            "so": "Somali",
            "es": "Spanish",
            "su": "Sundanese",
            "sw": "Swahili",
            "sv": "Swedish",
            "tg": "Tajik",
            "ta": "Tamil",
            "te": "Telugu",
            "th": "Thai",
            "tr": "Turkish",
            "uk": "Ukrainian",
            "ur": "Urdu",
            "ug": "Uyghur",
            "uz": "Uzbek",
            "vi": "Vietnamese",
            "cy": "Welsh",
            "xh": "Xhosa",
            "yi": "Yiddish",
            "yo": "Yoruba",
            "zu": "Zulu"
        }
        language_selection = st.selectbox("Select language:", list(languages.values()))
        if st.button("Translate"):
            target_language = next(key for key, value in languages.items() if value == language_selection)
            # Translate summary
            translated_text = translate_file_to_language(x, target_language)
            st.write(translated_text)

if __name__ == "__main__":
    main()
