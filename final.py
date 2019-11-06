import os
import math
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
import pandas as pd
import numpy as np
from tkinter import *
import random
from array import * 
import time
import binascii
from heapq import nlargest
from heapq import heapify,heappop,heappush





# os.chdir("C:/")
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\uni\\Data science\\songdata.csv')
df = df.drop(['link'], axis=1)

# getting all singers in a list
artistList = []
for artist in df['artist']:
    if artist not in artistList:
        artistList.append(artist)

WORD = re.compile(r'\w+')


# text_to_vector
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)


# get_cosine
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


# cleaning Text
def clearText(String):
    pattern = re.compile(r'\b(' + r'|'.join(
        stopwords.words('english')) + r')\b\s*')  # \b: Matches only the beginning or end of the word.
    text = re.sub(r'\W+', ' ', String)  # for symbol
    text = re.sub(r'\d+', ' ', text)  # digits
    text = pattern.sub(' ', text.lower())
    text = re.sub(r'\s+', ' ', text)  # for space
    tokenize = word_tokenize(text)
    return (tokenize)


# summarizedSong
def summarizedSong(singerName, n):
    lyrics = df.iloc[:, [0, -1]]
    lyrics = {k: ''.join(g["text"].tolist()) for k, g in lyrics.groupby("artist")}

    songs = str(lyrics[singerName])
    tokenize = clearText(songs)

    fdist = FreqDist(tokenize)
    max = fdist.most_common(1)

    # conver to weight
    for x in fdist:
        fdist[x] = round(fdist[x] / max[0][1], 2)

    # Product sentence scores
    sent2score = {}
    sent2score = {sentence: fdist[x] if sentence not in sent2score.keys() else sent2score[sentence] + fdist[x]
                  for sentence in songs.split('\n') for x in sentence.split(' ') if x in fdist.keys()}

    top_common = dict(Counter(sent2score).most_common(n)).keys()
    return ([*top_common])


# remix
def remix(artist):
    if artist not in artistList:
        return("The singer name isn't valid")
    singerList = artistList
    mainSong = summarizedSong(artist,5)
    singerList.remove(artist)

    list = []
    for x in artistList[0:100]:
        text = summarizedSong(x,5)
        list.append(text)

    cosValues = {}
    for sen in range(len(mainSong)):
        for text in range(len(list)):
            for sentence in range(len(list[text])):
                text1 = mainSong[sen]
                text2 = list[text][sentence]
                textt1 = text1.lower()
                text2 = text2.lower()

                vector1 = text_to_vector(text1)
                vector2 = text_to_vector(text2)
                cosine = get_cosine(vector1, vector2)
                cosValues[text2] = cosine

    songDict = Counter(cosValues).most_common(10)
    song = [x[0] for x in songDict]
    return (song)

#___________________________________________________________________________________________________

def shingle(total_text, shingle_size ):
    ShingleSets = {}
    shingleNo=0
    for docID, sentence in total_text.items():
        shinglesToInt = set()

        for word in range(len(sentence) - shingle_size + 1):
            shingle = sentence[word: word + shingle_size]
            shingle = ' '.join(shingle)

            crc = binascii.crc32(shingle.encode()) & 0xffffffff
            if crc not in shinglesToInt:
                shinglesToInt.add(crc)
                shingleNo = shingleNo + 1

            ShingleSets[docID]= shinglesToInt
    return ShingleSets, shingleNo


def findRandomNos(k, totalShingles):
  randList = []
  randIndex = random.randint(0, totalShingles -1) 
  randList.append(randIndex)
  while k>0:
    while randIndex in randList:
      randIndex = random.randint(0, totalShingles-1) 
      
    randList.append(randIndex)
    k = k-1
    
  return randList



def MinHash(sh_total ,Nsh):
        
    randomNoA = findRandomNos(25,Nsh)
    randomNoB = findRandomNos(25, Nsh)

    docLowestShingleID = {}
    for doc in sh_total.keys():
        shingleIDSet = sh_total[doc]
        signatures = []
        for x in range(0,25):
            listFx = []
            for shingleID in shingleIDSet:
              temp = (randomNoA[x] * shingleID + randomNoB[x]) % Nsh 
              listFx.append(temp)
            heapify(listFx) 
            signatures.append(heappop(listFx))
        docLowestShingleID[doc] = signatures

    return docLowestShingleID

def Jaccard_Similarities(docLowestShingleID , n):
    jaccarArray=[]
    for docID, lowest_ID in docLowestShingleID.items():
        shinglesSet1 = set(lowest_ID)
        jaccarArr={}
        for docID2, lowest_ID2 in docLowestShingleID.items():
            if docID2 > n:
                shinglesSet2 = set(docLowestShingleID[docID2])
                jaccard = (len(shinglesSet1.intersection(shinglesSet2)) / len(shinglesSet1.union(shinglesSet2)))
                jaccarArr[docID2] = jaccard
                # print(jaccard)
        jaccarArray.append(jaccarArr)
        if docID == n:
            break
    return jaccarArray







window = Tk()
window.geometry('700x550')
window.title("New Song Compilation")
#main layout
MainLabel = Label(window, text="Creating a New Song Compilation", justify=CENTER, fg="BLACK", font="Times 12 bold", width=65, height=2)
MainLabel.grid(row=1, columnspan=3)
label1 = Label(window,text="Insert a singer's name : ",fg="BLACK", font="Verdana 10 bold")
label1.grid(row=3,column=0)
SingerName = Entry(window, width=40)
SingerName.grid(row=3,column=1)
T = Text(window, height=20, width=40)

T.grid(row=7, column=1)


def running(event):
    T.delete('1.0', END)
    artist=remix(SingerName.get())

    T.insert(END, artist)

# SearchButton = Button(window, text="GO !", width=15, font="Helvetica 12 bold", height=1)
# SearchButton.bind('<Button-1>', running)

# SearchButton.grid(row=3, column=2)
# CheckBox1 = Checkbutton(window,text="r u human")
# CheckBox1.grid(row=2,column=0)
# frame1 = Frame(window,width=500, height=500,bg="RED")
# frame1.pack(side=BOTTOM)
# frame2 = Frame(window)
# frame2.pack(side=TOP,fill=Y)
# #label1 = Label(window, text="hiiiii",fg="GREEN",bg="RED")
# label2 = Label(frame2,text="hello",fg="GREEN",bg="RED")
# label2.pack(fill=Y)
# label1.pack(side=LEFT,fill=Y) #put in window
# button1 = Button(frame2,text="download",fg="RED")
# button2 = Button(frame2,text="remaining",fg="green")
# button3 = Button(frame1,text="cancel",fg="yellow")
# button4 = Button(frame1,text="resume")

# button1.pack(side=LEFT,fill=X)
# button2.pack(side=LEFT)
# button3.pack(side=RIGHT)
# button4.pack(side=RIGHT)

window.mainloop()



text_summarization=process(songs,7)
total_text={}
docID=0
for k,y in text_summarization.items():
    total_text[docID]=y
    docID = docID+1
for singer, songs in lyrics.items():
    if singer == singerName:
        continue
    for row in songs.split("\n"):
        if clearText(row) == []:
            continue
        total_text[docID]=clearText(row)
        docID=docID+1
        if docID == 30000:
            break
    else:
        continue
    break

sh_total, TNsh=shingle(total_text,2)
docLowestShingleID = MinHash(sh_total, TNsh)
JaccardSimilarities = Jaccard_Similarities (docLowestShingleID ,len(text_summarization) )

for x in JaccardSimilarities:
    most_similar = dict(Counter(x).most_common(1)).keys()
    print(most_similar)
    for s in most_similar:
        song = " ".join(total_text[s]) 
        print(song)
