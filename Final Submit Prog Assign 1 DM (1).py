
# Programming Assignment 1 -> Varad Nair | 1002161475

import os
import math
import nltk
nltk.download()
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def calculateIDF(token):
    """Calculate Inverse Document Frequency (IDF) for a given token."""
    total_docs = len(stemmed_speeches)
    token_count = doc_frequencies.get(token, 0)
    return -1 if token_count == 0 else math.log10(total_docs / token_count)


def getweight(document, term):
# getweight(filename,token): return the normalized TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, return 0. You should stem the parameter 'token' before calculating the tf-idf score.    
    """Calculate the weight of a term in a document."""
    
    if document not in stemmed_speeches:
        print("Invalid Document name entered so returning minus 2 below")
        return -2  # Indicating an error condition

    term = stemmer.stem(term)
    term_freq = stemmed_speeches[document].count(term)
    
    # Proceed with the original logic if the document name is valid
    if term_freq == 0:
        return 0
    else:
        return (1 + math.log10(term_freq)) * getidf(term) / magnitudes[document]
    '''
    term = stemmer.stem(term)
    term_freq = stemmed_speeches[document].count(term)
    return 0 if term_freq == 0 else ((1 + math.log10(term_freq)) * getidf(term)) / magnitudes[document]
    '''
def query(query_str):
# query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect to the weighting scheme. You should stem the query tokens before calculating similarity.

    """Process and evaluate a query against the corpus."""
    processed_query = processQuery(query_str)
    normalized_query = calculateQueryVector(processed_query)

    max_similarity = 0
    best_match = ()

    for file, norm_speech in normalized_speeches.items():
        similarity = sum(normalized_query.get(token, 0) * norm_speech.get(token, 0) for token in normalized_query)
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = (file, max_similarity)

    return best_match if best_match else ('No Document', -1)

def getidf(token):
# getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. You should stem the parameter 'token' before calculating the idf score.    
    token = stemmer.stem(token)
    return calculateIDF(token)

def processQuery(query):
    #Tokenizing, filtering stopwords, and stemming a query.
    query = query.lower()
    tokens = tokenizer.tokenize(query)
    filtered = [word for word in tokens if word not in stop_words]
    return [stemmer.stem(word) for word in filtered]

def calculateQueryVector(query):
    """Calculate the vector representation of a query."""
    term_freq = {word: query.count(word) for word in query}
    weighted_tf = {word: 1 + math.log10(freq) for word, freq in term_freq.items()}
    magnitude = math.sqrt(sum(value ** 2 for value in weighted_tf.values()))
    return {word: value / magnitude for word, value in weighted_tf.items()}


# Initialize directory path for the corpus of US Inaugural Addresses
corpus_dir = './US_Inaugural_Addresses'
speeches_dict = {}

# Read and preprocess the speeches from the corpus
for speech_file in os.listdir(corpus_dir):
    if speech_file.startswith(('0', '1', '2', '3')):
        with open(os.path.join(corpus_dir, speech_file), "r", encoding='windows-1252') as file:
            speeches_dict[speech_file] = file.read().lower()

# Initialize tokenizer, stopwords, and stemmer for preprocessing
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenize, remove stopwords, and stem the speeches
tokenized_speeches = {file: tokenizer.tokenize(text) for file, text in speeches_dict.items()}
filtered_speeches = {file: [word for word in tokens if word not in stop_words] for file, tokens in tokenized_speeches.items()}
stemmed_speeches = {file: [stemmer.stem(word) for word in tokens] for file, tokens in filtered_speeches.items()}

# Calculate document frequencies for each token
doc_frequencies = {}
for file, tokens in stemmed_speeches.items():
    for token in set(tokens):
        doc_frequencies[token] = doc_frequencies.get(token, 0) + 1



#Applying Normalization:-> Normalize speeches and calculate their magnitudes
normalized_speeches = {}
magnitudes = {}


for file, tokens in stemmed_speeches.items():
    term_freq = {word: tokens.count(word) for word in tokens}
    weighted_terms = {word: 1 + math.log10(freq) for word, freq in term_freq.items()}
    weighted_idf = {word: weight * calculateIDF(word) for word, weight in weighted_terms.items()}
    magnitude = math.sqrt(sum(value ** 2 for value in weighted_idf.values()))

    if magnitude == 0:
        raise Exception('Division by zero encountered due to all magnitudes being 0.')

    normalized_speech = {word: value / magnitude for word, value in weighted_idf.items()}
    normalized_speeches[file] = normalized_speech
    magnitudes[file] = magnitude


print("%.12f" % getidf('children'))
print("%.12f" % getidf('foreign'))
print("%.12f" % getidf('people'))
print("%.12f" % getidf('honor'))
print("%.12f" % getidf('great'))
print("--------------")
# Handling Edge Case like if wrong file name entered with Try Except. So put the values in between those block
try:
    print("%.12f" % getweight('19_lincoln_1861.txt', 'constitution'))
    print("%.12f" % getweight('23_hayes_1877.txt', 'public'))
    print("%.12f" % getweight('25_cleveland_1885.txt', 'citizen'))
    print("%.12f" % getweight('09_monroe_1821.txt', 'revenue'))
    print("%.12f" % getweight('05_jefferson_1805.txt', 'press'))
except ValueError as e:
    print(e)

print("--------------")
print("(%s, %.12f)" % query("pleasing people"))
print("(%s, %.12f)" % query("war offenses"))
print("(%s, %.12f)" % query("british war"))
print("(%s, %.12f)" % query("texas government"))
print("(%s, %.12f)" % query("cuba government"))
