import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import spacy
from transformers import pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load English language model for SpaCy
nlp = spacy.load('en_core_web_sm')

# Load summarization pipeline from transformers library
summarizer = pipeline("summarization")

def preprocess_text(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    
    # Remove stopwords and punctuation from each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stopwords.words('english')]
        preprocessed_sentences.append(' '.join(words))
    
    return preprocessed_sentences

def summarize_text(text):
    # Preprocess the text
    preprocessed_sentences = preprocess_text(text)
    
    # Join preprocessed sentences into a single string
    preprocessed_text = ' '.join(preprocessed_sentences)
    
    # Summarize the preprocessed text
    summary = summarizer(preprocessed_text, max_length=100, min_length=20, do_sample=False)[0]['summary_text']
    
    # Tokenize the summary into sentences
    summary_sentences = sent_tokenize(summary)
    
    # Ensure each sentence has meaningful content
    meaningful_summary = []
    for sentence in summary_sentences:
        doc = nlp(sentence)
        if len(doc) > 1:  # Ensure sentence has more than one token
            meaningful_summary.append(sentence)
    
    return meaningful_summary

# Example usage
text = """
William Yarnel Slack (August 1, 1816 - March 21, 1862) was an American lawyer, politician, and military officer who fought for the Confederate States of America during the American Civil War. Born in Kentucky, Slack moved to Missouri as a child and later entered the legal profession. After serving in the Missouri General Assembly from 1842 to 1843, he fought as a captain in the United States Army for fourteen months during the Mexican-American War, beginning in 1846. He saw action at the Battle of Embudo Pass and the Siege of Pueblo de Taos. Returning to a legal career, Slack became influential in his local area.

After the outbreak of the American Civil War in April 1861, Slack, who held pro-slavery views, supported the Confederate cause. When the Missouri State Guard (MSG) was formed the next month to oppose the Union Army, he was appointed as a brigadier general in the MSG's 4th Division. After participating in the Battle of Carthage in July, he fought in the Battle of Wilson's Creek on August 10. After a surprise Union attack, Slack's deployment of his division gave time for further Confederate States Army and MSG troops to deploy. Suffering a bad hip wound at Wilson's Creek, he was unable to rejoin his command until October.

Along with other Missouri State Guard officers, Slack transferred to the Confederate States Army in late 1861 where he commanded a brigade with the rank of colonel. On March 7, 1862, during the Battle of Pea Ridge, Slack suffered another wound that was close to the injury he had received at Wilson's Creek. Infection set in, and he died on March 21. He was posthumously promoted to brigadier general in the Confederate army on April 17; the Confederate States Senate might not have known that he was dead at the time of the promotion.
"""

summary = summarize_text(text)
for sentence in summary:
    print(sentence)
