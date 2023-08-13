import random
import json
import numpy as np
import pickle
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# accessing the key intents of the object intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)  # getting a text, and splits up into words
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:  # check if the tag is already in the class
            classes.append(intent['tag'])

# print(documents)

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))  # ignore the duplicates
classes = sorted(set(classes))
# print(words)

# save them in a file in binaries
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# we need to represent the words in numerical values, neural network needs numerical values, we use bag-of-words
training = []
outputEmpty = [0] * len(classes)

# in this loop, all the document data goes in the training list
for document in documents:
    bag = []
    wordPatterns = document[0]  # Fix: Use the current document's word list
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)  # copying the list
    outputRow[classes.index(document[1])] = 1
    training.append([bag, outputRow])

random.shuffle(training)
# Split the training list into trainX and trainY
trainX = np.array([item[0] for item in training])
trainY = np.array([item[1] for item in training])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))  # to prevent overfitting
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

# Use the legacy optimizer
sgd = tf.keras.optimizers.legacy.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Model Saved!")
