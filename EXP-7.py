import nltk
from nltk.util import ngrams
from nltk.probability import FreqDist

# Download required package
nltk.download('punkt')
nltk.download('punkt_tab')

# -----------------------------
# Input Tweet
# -----------------------------
tweet = input("Enter a tweet: ")

# Tokenization
tokens = nltk.word_tokenize(tweet.lower())

print("\nTokens:")
print(tokens)

# -----------------------------
# N-Gram Model
# -----------------------------
print("\nUnigrams:")
print(list(ngrams(tokens, 1)))

print("\nBigrams:")
print(list(ngrams(tokens, 2)))

print("\nTrigrams:")
print(list(ngrams(tokens, 3)))

# Word Frequencies
fd = FreqDist(tokens)

print("\nWord Frequencies:")
for word, freq in fd.items():
    print(word, ":", freq)

# -----------------------------
# HMM Prediction (Sample)
# -----------------------------
print("\nHMM Prediction (Sample)")

sample_sentence = [
    ("AI", "NOUN"),
    ("improves", "VERB"),
    ("technology", "NOUN")
]

for word, tag in sample_sentence:
    print(word, "->", tag)

# -----------------------------
# Comparison
# -----------------------------
print("\nComparison")

print("N-Gram Model:")
print("- Learns word sequences.")
print("- Predicts the next word based on previous words.")
print("- Used for language modeling.")

print("\nHidden Markov Model (HMM):")
print("- Predicts word tags.")
print("- Uses transition probabilities.")
print("- Used for sequence labeling.")