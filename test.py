import nltk
import csv
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
set(stopwords.words('english'))
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
ps = PorterStemmer()
print("Enter the number of topics")
tp=int(input())
tparray=[]
for i in range(tp):
    tparray.append(i)
#eg=pd.read_csv("abcnews-date-text.csv", header =[1, 2])
#print(ds)
stop_words = set(stopwords.words('english'))#Loading English Stopwords
file1 = open("test.csv", "r")
line = file1.read()# Use this to read file content as a stream:
words = line.split()
cnt=0
for r in words:
    r=r.lower()
    if(r[:3] == "doc"):
        cnt+=1
list=[[] for i in range(cnt)]#Creating 2d empty list
doctop=[[] for q in range(cnt)]
i=-1
for r in words:
    if not r in stop_words:
        r=r.lower()#Lower Case the Words
        r = ps.stem(r)#Stemming the words
        if (r[:3] == "doc"):
            i+=1
        if(r[:3] =="doc"):
            p=r.find(',')
            r=r[p+1:]
        list[i].append(r)#Appending at the end of list
pp=0
for i in range(len(list)):
    for j in range(len(list[i])):
        pp+=1
        x = np.random.randint(0, tp)#Used for random assignment of topics
        doctop[i].append(x)
m_set=set([])#Set DS used for getting distinct values
#print("2d List representing random assignment of topics")
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        m_set.add(list[i][j])#Add each word in the set
 #       print(doctop[i][j], end=" ")
#    print("\n")
arr=np.full((cnt, tp), 0)#Used to create 2d numpy array
#arr array will correspond to frequency of each topic in each document, rows-documents, cols-topics
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        arr[i][doctop[i][j]]+=1
#print("2d numpy array representing frequency of topics in documents")
#print(arr)
tpword=np.full((tp, len(m_set)), 0)#2d numpy array which will correspond to frequency of each word correspoding to each topic
Dict={}#Dictionary used to map words or to create associative arrays
c=0
for val in m_set:
    Dict[val]=c#Adding a word in Dictionary
    c+=1
for i in range(len(doctop)):
    for j in range(len(doctop[i])):
        tpword[doctop[i][j]][Dict[list[i][j]]]+=1#Used to increment frequency count
        #rows represent topic, cols represents words
#print("2d numpy array representing frequency of words in each topic")
#print(tpword)
"""print(np.sum(tpword))
print(len(m_set))
print(pp)"""
#prarr=np.full((cnt, tp), 0)
prarr=[[] for q in range(cnt)]
for i in range(len(arr)):
    s=0
    for j in range(len(arr[i])):
        s+=arr[i][j]
    for j in range(len(arr[i])):
        v=arr[i][j]/s #to find probabbility in random assignment
        #prarr[i][j]=v
        if v==0:
            v=1/(s+(len(arr[i])+1) ) #if the topic belongs to doc 0, then it's probability is given by(1/sum of number of topics assigned in a doc*total no. of docs)
        prarr[i].append(v)

prtpwrd=[[] for q in range(tp) ]
for i in range(len(tpword)):
    s=0
    for j in range(len(tpword[i])):
        s+=tpword[i][j]
    for j in range(len(tpword[i])):
        v=tpword[i][j]/s
        if v==0:
            v=1/(s+(len(tpword[i]) +1))  #if the word belongs to topic 0, then it's probability is given by(1/sum of number of words assigned in a topics*total no. of topics)
        prtpwrd[i].append(v)

print("Before LDA")
for i in range(len(prarr)) :
    for j in range(len(prarr[i])) :
        if(j!=len(prarr[i])) :
            print(np.array(prarr[i][j]),end=' ')
    print('\n')
L11=[]
L12=[]
L13=[]
L21=[]
L22=[]
L23=[]
for i in range(10) :
    L11.append(prarr[i][0])
for i in range(10) :
    L12.append(prarr[i][1])
for i in range(10) :
    L13.append(prarr[i][2])
with open('C:\\Users\\Harry\\PycharmProjects\\untitled1\\venv\\BLDA.csv', 'w') as writeFile:
    writeFile.truncate()
    writer = csv.writer(writeFile)

    writer.writerows(prarr)



#print(prarr)
writeFile.close()
for p in range(100):
    chs=[None]*tp
    for i in range(len(list)):
        for j in range(len(list[i])):
            for k in range(tp):
                chs[k]=prarr[i][k]
                chs[k]=chs[k]*prtpwrd[k][Dict[list[i][j]]]
            sum=0
            for k in range(tp):
                sum+=chs[k]
            for k in range(tp):
                chs[k]=chs[k]/sum
            #print(chs)
            pre=doctop[i][j]
            post=int(np.random.choice(tparray, 1, chs))
            doctop[i][j]=post
            arr[i][pre]=max(0, arr[i][pre]-1)
            arr[i][post]+=1
            tpword[pre][Dict[list[i][j]]]=max(0, tpword[pre][Dict[list[i][j]]]-1)
            #print(Dict[list[i][j]], post)
            tpword[post][Dict[list[i][j]]]+=1
            s=0
            for k in range(len(arr[i])) :
                s+=arr[i][k]
            for k in range(len(arr[i])) :
                v=arr[i][k]/s
                if v == 0:
                    v = 1 / (s + (len(arr[i]) + 1))
                prarr[i][k]=v
            s=0
            for k in range(len(prtpwrd[pre])) :
                s+=tpword[pre][k]
            for k in range(len(prtpwrd[pre])):
                v=tpword[pre][k]/s
                if v == 0:
                    v = 1 / (s + (len(prtpwrd[pre]) + 1))
                prtpwrd[pre][k]=v
            s = 0
            for k in range(len(prtpwrd[post])):
                s += tpword[post][k]
            for k in range(len(prtpwrd[post])):
                v = tpword[post][k] / s
                if v == 0:
                    v = 1 / (s + (len(prtpwrd[post]) + 1))
                prtpwrd[post][k] = v
print("After LDA")
for i in range(len(prarr)) :
    for j in range(len(prarr[i])) :
        if(j!=len(prarr[i])) :
            print(np.array(prarr[i][j]),end=' ')
    print('\n')

with open('LDA.csv', 'w') as writeFile:
   # spamreader = csv.reader(csvfile, delimiter= 'A', quotechar='|')
    writer = csv.writer(writeFile)

    writer.writerows(prarr)

writeFile.close()
for i in range(10) :
    L21.append(prarr[i][0])
for i in range(10) :
    L22.append(prarr[i][1])
for i in range(10) :
    L23.append(prarr[i][2])


# data to plot
n_groups = 10


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, L11, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Topic 0')

rects2 = plt.bar(index + bar_width, L12, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Topic 1')
rects3 = plt.bar(index + bar_width+ bar_width, L13, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Topic 2')
ll=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xlabel('Documents')
plt.ylabel('Probability')
plt.title('Before LDA')
plt.xticks(index + bar_width, ll)
plt.legend()

plt.tight_layout()
plt.show()




# data to plot
n_groups = 10


# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8

rects1 = plt.bar(index, L21, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Topic 0')

rects2 = plt.bar(index + bar_width, L22, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Topic 1')
rects3 = plt.bar(index + bar_width +bar_width, L23, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Topic 2')
ll=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
plt.xlabel('Documents')
plt.ylabel('Probability')
plt.title('After LDA')
plt.xticks(index + bar_width, ll)
plt.legend()

plt.tight_layout()
plt.show()