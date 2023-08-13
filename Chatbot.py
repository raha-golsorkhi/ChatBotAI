import random
import numpy as np
import pickle
import json
import tensorflow as tf
import nltk 
from nltk.stem import WordNetLemmatizer

#from tensorflow import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

#to read in binary mode
words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = tf.keras.models.load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#convert a sentence into a bag of words, list full of 0s, flags saying if the word exist or no
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words) #initial bag of zeros
    for w in sentence_words:
        for i ,word in enumerate(words):
            if word == w :
                bag[i] = 1
    return np.array(bag) #bag of words


#uses 2 functions above to do prediction
def predict_class(sentence):
    bow = bag_of_words(sentence) #feed into neuro network
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = .25
    result = [[i, r]for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    result.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})

    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break

    return result

print("BOT is running")

while True:
    message = input("")
    ints = predict_class(message)
    if not ints:
        print("Sorry, I don't understand that.")
    else:
        res = get_response(ints, intents)
        print(res)

