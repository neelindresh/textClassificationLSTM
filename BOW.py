import nltk
from nltk.corpus import stopwords
import pandas as pd
import re
import CreateVocab
from model import Recommndation
import torch
import tqdm
#nltk.download('stopwords')
words= stopwords.words('english')
df=pd.read_csv('consolidated.csv')
df=df[['name','tag','title',"watched"]]

def clear_name(name):
    name=re.sub("[^A-Za-z]","",name)
    return name

def create_title(name):
    name=name.lower()
    name=re.sub("[^A-Za-z ]","",name)
    name=[i for i in name.split() if i not in words]
    return name

def tags(taglist):
    taglist=eval(taglist)
    taglist=" ".join(taglist).lower()
    taglist=re.sub("[^A-Za-z ]","",taglist)
    return taglist.split()

def createBOW(df):
    df["name"] = df["name"].apply(clear_name)
    df["title"]=df["title"].apply(create_title)
    df["tag"]=df["tag"].apply(tags)
    BOW=[]
    for name,tag,title,label in df.values:
       BOW.append({'bow':" ".join([name]+tag+title),"lable":label})
    return BOW

def word2tensor(vocab,sentence):
    index_words=[]
    padding=[]
    for i in sentence:
        index_words.append(vocab.word2index[i])
    if len(index_words)<vocab.maxwords:
        padding=[0 for i in range(vocab.maxwords-len(index_words))]
    index_words+=padding
    return torch.LongTensor(index_words)

bow_data=createBOW(df)
df=pd.DataFrame(bow_data)
df.to_csv("classfication_data.csv",index=False)
'''
vocab=CreateVocab.Vocab()
for i in bow_data:
    vocab.add_word(i['bow'])
#print(vocab.nwords)
rec=Recommndation(vocab.nwords,vocab.maxwords)
optimizer=torch.optim.Adam(rec.parameters())
criterion=torch.nn.BCELoss()


for i in range(10):
    lossperepoch=0
    for word in tqdm.tqdm(bow_data):
        optimizer.zero_grad()
        inp=word2tensor(vocab,word['bow'])
        target=torch.FloatTensor([0.]) if word['lable']==False else torch.tensor([1.])
        output=rec(inp)
        output=output.type(torch.FloatTensor)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        lossperepoch+=loss.item()
    print("epoch ",i, "| Loss :",lossperepoch)
'''
