from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')

app = Flask(__name__)

# Dummy corpus of documents
documents = [
    "This is a sample document.",
    "This document is another example.",
    "Here is some more sample text."
]

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return ' '.join(tokens)

def check_plagiarism(input_text):
    input_text = preprocess(input_text)
    corpus = [preprocess(doc) for doc in documents] + [input_text]

    vectorizer = TfidfVectorizer().fit_transform(corpus)
    vectors = vectorizer.toarray()
    cosine_matrix = cosine_similarity(vectors)
    
    input_vector = cosine_matrix[-1][:-1]
    highest_score = max(input_vector)
    highest_index = input_vector.tolist().index(highest_score)
    
    return highest_score, documents[highest_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check():
    text = request.form['text']
    plagiarism_percentage, matched_text = check_plagiarism(text)
    plagiarism_percentage = round(plagiarism_percentage * 100, 2)
    return render_template('result.html', text=text, plagiarism_percentage=plagiarism_percentage, matched_text=matched_text)

if __name__ == '__main__':
    app.run(debug=True)
