#!/usr/bin/env python
# coding: utf-8

# In[53]:


import tensorflow as tf
import keras
from keras.models import load_model
import tensorflow_addons as tfa
from keras_preprocessing.sequence import pad_sequences
from nltk.tokenize import word_tokenize


# In[54]:


import nltk
nltk.download('punkt')


# In[55]:


import os
os.environ['JAVAHOME'] =  "C:\Program Files\Java\jdk-18.0.2"


# In[56]:


from distutils import text_file
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)


# In[57]:


results = []
count = []
def detect_sentiment():
    text = textArea.get("1.0","end")
    #file_in = open(filename, 'r')
    lines = text.split('.')
    lines.pop()
    count1, count2, count3 = 0,0,0
    for line in lines:
        #sentence = textArea.get("1.0", "end")
        sa = SentimentIntensityAnalyzer()
        sentiment_dict = sa.polarity_scores(line)
        #sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05 :
            string = " Positive "
            count1+=1
 
        elif sentiment_dict['compound'] <= - 0.05 :
            string = " Negative "
            count2+=1
      
        else :
            string = " Neutral "
            count3+=1
        results.append(string)
        result_area.insert(END, string+'\n')
    count.append(count1)
    count.append(count2)
    count.append(count3)


# In[58]:


def open_txt():
    text_file = filedialog.askopenfilename(initialdir="C:/Users/HP/Desktop/Project/", title="Select a file", filetypes=[('text files','*.txt')])
    text_file = open(text_file, 'r')
    stuff = text_file.read()
    textArea.insert(END, stuff)
    #text_file.close()


# In[59]:


def ner_prediction():
    model = load_model('ner_model.h5', custom_objects={"crf":tfa.layers.CRF})
    tags = ['B-gpe', 'I-org', 'I-gpe', 'I-tim', 'O', 'I-art', 'I-nat', 'I-eve', 'I-geo', 'B-art', 'B-per', 'B-org', 'B-eve', 'B-geo', 'B-nat', 'I-per', 'B-tim']
    text = textArea.get("1.0","end")
    words = list(set(text.split())) 
    word2idx = {w: i + 2 for i, w in enumerate(words)}
    word2idx["UNK"] = 1 # Unknown words
    word2idx["PAD"] = 0 # Padding
    idx2word = {i: w for w, i in word2idx.items()}
    max_len = 75
    x = {}
    
    pad_line = pad_sequences(sequences=[[word2idx.get(w, 0)] for w in word_tokenize(text)], padding="post", value=0, maxlen=max_len)
    for i in range(len(pad_line)):
        p = model.predict(np.array([pad_line[i]]))
        p = np.argmax(p, axis=-1)
        for w, pred in zip(pad_line[i], p[0]):
            x[w] = tags[pred]
    y = {}
    for num in list(x.keys()):
        y[idx2word[num]] = x[num]
                
    result_area.insert(END, y)


# In[ ]:





# In[60]:


#ner_res = []
def stanford_ner():
    from nltk.tag import StanfordNERTagger
    from nltk.tokenize import word_tokenize
    st = StanfordNERTagger('D:\stanford-ner-4.2.0\stanford-ner-2020-11-17\classifiers\english.all.3class.distsim.crf.ser\english.all.3class.distsim.crf.ser',
                          'D:\stanford-ner-4.2.0\stanford-ner-2020-11-17\stanford-ner.jar',
                          encoding='utf-8')
    nltk.download('punkt')
    text = textArea.get("1.0","end")
    lines = text.split('.')
    lines.pop()
    for line in lines:
        tokenized_text = word_tokenize(text)
        classified_text = st.tag(tokenized_text)
        #ner_res.append(classified_line)
        result_area.insert(END, classified_text)


# In[61]:


def plot_it():
    fig = Figure(figsize = (2, 2), dpi = 100)
    fig.add_subplot(111).pie(count)
    canvas = FigureCanvasTkAgg(fig, master = gui)  
    canvas.draw()
    canvas.get_tk_widget().grid(row = 5, column = 1, pady = 2)
    #toolbar = NavigationToolbar2Tk(canvas, gui)
    #toolbar.update()
    #canvas.get_tk_widget().grid(row = 5, column = 1, pady = 2)
    #plt.figure()
    #plt.pie(count)
    #plt.legend(['Positive', 'Negative', 'Neutral'])
    #plt.show(block = False)
    
def clear_all():
    textArea.delete('1.0', END)
    result_area.delete('1.0', END)

def save_file():
    saved_file = open('C:/Users/HP/Desktop/Project/results.txt', 'w')
    saved_file.write(result_area.get())


# In[62]:


if __name__ == "__main__":
    gui = tk.Tk()
    gui.geometry("700x700")
    gui.title("NLP App")
    gui.configure(bg="purple")
    result_var = tk.StringVar()
    textArea = Text(gui, width = 40, height = 5, font=("Arial", 16))
    textArea.grid(row = 0, column = 1, pady = 2)
    open_button = Button(gui, text = "Open text file", command = open_txt)
    open_button.grid(row = 1, column = 1, pady = 2)
    check_sentiment = Button(gui, text = "Check Sentiment", command=detect_sentiment)
    check_sentiment.grid(row = 2, column = 0, pady = 2)
    check_ner1 = Button(gui, text = "Apply Custom NER", command=ner_prediction)
    check_ner1.grid(row = 2, column = 1, pady = 2)
    check_ner = Button(gui, text = "Apply Stanford NER", command=stanford_ner)
    check_ner.grid(row = 2, column = 2, pady = 2)
    result_area =  Text(gui, width = 40, height = 5, font=("Arial", 16))
    result_area.grid(row = 3, column = 1, pady = 2)
    plotgraph1 = Button(gui, text = "Graph Sentiment prediction", command = plot_it)
    plotgraph1.grid(row = 4, column = 1,pady = 2)
    save_button = Button(gui, text = "Save file", command= save_file)
    save_button.grid(row = 6, column = 1, pady = 2)
    clear_button = Button(gui, text = "Clear", command= clear_all)
    clear_button.grid(row = 7, column = 1, pady = 2)
    exit_button = Button(gui, text = "Exit", command = exit)
    exit_button.grid(row = 8, column = 1, pady = 2)
    gui.mainloop()


# In[ ]:





# In[ ]:





# In[ ]:




