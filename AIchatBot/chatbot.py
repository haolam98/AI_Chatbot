import tensorflow
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import random
import json
import pickle

#read in json file
with open("intents.json") as file:
    data= json.load(file)

try:
    # open model file
    with open("data.pickle","rb") as f:
        words,labels,training,output= pickle.load(f)
except:
    #looping through the data and classify data for dictionary
    words = []
    labels = []
    docs_x = []
    doccs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            #getting the root of the word - main meaning of the word
             #make the model more accurate

            #get all the word with tokenize
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds) # add all words in

            docs_x.append(wrds)
            doccs_y.append(intent["tag"])

            #get all the tags to labels
            if intent["tag"] not in labels:
                labels.append(intent["tag"])

    #stem all the words and remove the duplicate
    words = [stemmer.stem(w.lower()) for w in words if w!="?"] #remove any '?' in the text
        #'set' will remove duplicate, then 'list' will list all of them again, then sort it
    words = sorted(list(set(words)))
    labels=sorted(labels)

    # Create a bad of words - encode if the word is in the sentence and how many time
    '''
    Ex: [1,0,0,1,1,1,0,0....]  
        ['a','bite','the','is',...]
        => each number represent if the word appear in the text
        => So 'a' appears in the text
              'bite' is not appears 
              'is' appears in the text
              ...
    '''
    training =[]
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x,doc in enumerate(docs_x):
        bag =[]
        #stem words
        wrds = [stemmer.stem(w.lower()) for w in doc]

        #go through all the word in the document that are now stemed into bag[]
            #we going to put 0 or 1 for if it's in our main 'words' list above
        for w in words:
            if w in wrds:
                bag.append(1) #this word exists
            else:
                bag.append(0) #not exist
        output_row = out_empty[:]

        #look through the labels[] and see where the tag is in that list, and set it to 1 in the ouput[]
        output_row[labels.index(doccs_y[x])]=1

        training.append(bag)
        output.append(output_row)

    #change training and ouput to numpy array so that we can feed them into our model
    training = numpy.array(training)
    output = numpy.array(output)

    #save file
    with open("data.pickle","wb") as f:
        pickle.dump((words,labels,training,output),f)


#get rid of any pre-setting
#tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8) #add fully connected network of hidden layer of 8 neuron
net = tflearn.fully_connected(net,8) #another hidden layer of 8 neuron
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

    #-DNN: type of neuro network
model = tflearn.DNN(net)

# train and save model
try:
     model.load("model.tflearn")
except:
#n_epoch: number of time the machine going to see the data
    with tensorflow.Session() as sess:
        model.fit(training,output,n_epoch=1000, batch_size=8, show_metric=True)
        model.save("model.tflearn")


#turn user's input to bad of word

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words= [stemmer.stem(word.lower()) for word in s_words]

    #generate bag list
    for se in s_words:
        for i,w in enumerate(words):
            #if current word in words[] == the word in the sentence
            if w ==se:
                bag[i]=1
    return numpy.array(bag)


#ask user for input

def chat():
    print("Please talk with the bot! Type 'quit' to end")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            print("Bot: I see we had a good talk. So sad it ends...\n--SESSION END--")
            break

        result = model.predict([bag_of_words(inp,words)])[0]

        result_index = numpy.argmax(result)#return the index of answer has highest probability
        tag = labels[result_index]

        if result[result_index] >0.7:

            #go through data in json file to get respond
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print("Bot: ",random.choice(responses))
        else:
            print("Sorry, I didn't get that. Please try another question")

chat()
