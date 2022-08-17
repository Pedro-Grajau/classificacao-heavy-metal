from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

lemmatizer = WordNetLemmatizer()
vectorizer = pickle.load(open('modelos_salvos/tfidf_vectorizer.sav', 'rb'))

class TextProcessing:

    def __init__(self, lyrics):
        self.lyrics = lyrics
    
    def to_lower(self):
        self.lyrics = self.lyrics.lower()
    
    def remove_stopwords(self):
        tokenized = word_tokenize(self.lyrics)
        stop_words = set(stopwords.words('english'))
        filtered_stop = [word for word in tokenized if word not in stop_words]
        self.lyrics = " ".join(filtered_stop)

    def remove_punctuation(self):
        tokenized = word_tokenize(self.lyrics)
        no_punctuation = [word for word in tokenized if word.isalpha()]
        self.lyrics = " ".join(no_punctuation)

    def word_lemmatizer(self):
        tokenized  = word_tokenize(self.lyrics)
        lemmatized = [lemmatizer.lemmatize(word) for word in tokenized]
        self.lyrics = " ".join(lemmatized)

    def vectorize(self):
        return vectorizer.transform([self.lyrics]).toarray()
    
    def format_text(self):
        self.to_lower()
        self.remove_stopwords()
        self.remove_punctuation()
        self.word_lemmatizer()
        return self.vectorize()
