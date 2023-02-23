import json
import random
import string

def generate_sentence():
    # define the range of sentence length
    min_length = 5
    max_length = 20

    # define the range of word length
    min_word_length = 2
    max_word_length = 10

    # generate a random sentence
    words = []
    for i in range(random.randint(min_length, max_length)):
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(min_word_length, max_word_length)))
        words.append(word)
    sentence = ' '.join(words).capitalize() + '.'

    return sentence

# generate a list of random sentences
x = 404  # specify the number of sentences to generate
random_sentences = [generate_sentence() for _ in range(x)]

# save the list to a JSON file
with open('test_paragraphs.json', 'w') as f:
    json.dump(random_sentences, f)