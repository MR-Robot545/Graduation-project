import nltk
from nltk.corpus import brown
import random
import string


# Load Brown Corpus
nltk.download('brown')
sentences = brown.sents()


# Flatten the list of sentences into a single list of words
words = [word.lower() for sent in sentences for word in sent if word.isalnum()]

# Preprocess the words
punctuation = set(string.punctuation)
words = [word for word in words if word not in punctuation]


# Function to generate n-grams
def generate_ngrams(n, words):
    ngrams = {}
    for i in range(len(words)-n+1):
        ngram = tuple(words[i:i+n-1])
        next_word = words[i+n-1]
        if ngram in ngrams:
            ngrams[ngram].append(next_word)
        else:
            ngrams[ngram] = [next_word]
    return ngrams


# Build bi-gram and tri-gram models
bi_grams = generate_ngrams(2, words)
tri_grams = generate_ngrams(3, words)


# Function to generate a sentence
def generate_sentence(n, ngrams, max_len=20):
    start = random.choice(list(ngrams.keys()))
    sentence = list(start)
    while len(sentence) < max_len:
        if tuple(sentence[-(n-1):]) in ngrams:
            next_word = random.choice(ngrams[tuple(sentence[-(n-1):])])
            sentence.append(next_word)
        else:
            break
    return ' '.join(sentence)


# Main function to generate M sentences
def generate_sentences(m, n, ngrams, max_len=20):
    sentences = []
    for _ in range(m):
        maxx=random.randint(1, max_len)
        sentence = generate_sentence(n, ngrams, max_len)
        sentences.append(sentence)
    return sentences


# Parameters
m = int(input ("enter the number of Sentance: "))
n = int(input("enter the number of n-gram: "))
max_len = int(input("Max number of words in a sentence: "))

# Generate sentences
if n == 2:
    sentences = generate_sentences(m, n, bi_grams, max_len)
elif n == 3:
    sentences = generate_sentences(m, n, tri_grams, max_len)

# Print generated sentences
for i, sentence in enumerate(sentences, 1):
    print(f"Sentence {i}: {sentence}")