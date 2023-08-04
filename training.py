import random
import json
import numpy as np
import pickle
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer

lemmaizer = WordNetLemmatizer()

intents = json.load(open('intent.json'.read()))

words = []
classes = []
documents = []
ignorLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)
        documents.append(wordList, intent['tag'])
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignorLetters]
words = sorted(set(classes))   

classes = sorted(set(classes))

pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes.open('classes.pk1', 'wb'))

training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = []
    wordPatterns = documents[0]
    wordPatterns = [lemmaizer.lemmatize(word.lower()) for word in wordPatterns]
    for word in words: bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)
    outputRow[classes.indexdocument[1]] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

model = tf.keras.Sequential()

model.add(tf.keras.layer.Dense(128, input_shape = (len(trainX[0]),),activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation = 'softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.09, nesterov=True)

model.compile(loss = 'catagorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])
hist = model.fit(np.array(trainX), np.array(trainY), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot_simplelearnmodel.h5', hist)
print("Executed")
