import nltk
from nltk import word_tokenize, pos_tag

# Download required resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Get input from user
text = input("Enter legal text: ")

# Tokenize the text
tokens = word_tokenize(text)

# Perform POS tagging
tags = pos_tag(tokens)

print("\nDetected Named Entities:")

count = 0

# Detect Proper Nouns (NNP)
for word, tag in tags:
    if tag == "NNP":
        print(word, "-> ENTITY")
        count += 1

# Get actual number of entities
actual = int(input("\nEnter actual number of entities: "))

# Calculate accuracy
if max(count, actual) == 0:
    accuracy = 100
else:
    accuracy = (min(count, actual) / max(count, actual)) * 100

print("\nPredicted Entities:", count)
print("NER Accuracy:", round(accuracy, 2), "%")