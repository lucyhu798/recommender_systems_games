#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import numpy as np
import string
import random
from random import randint
import string
from sklearn import linear_model
from sklearn.svm import SVC


# In[2]:


def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N


# In[3]:


def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)


# In[4]:


def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d


# In[5]:


answers = {}


# In[6]:


allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)


# In[7]:


hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]


# In[8]:


ratingsPerUser = defaultdict(list)
ratingsPerGame = defaultdict(list)
allratingsPerUser = defaultdict(set)
allratingsPerGame = defaultdict(set)
randomGame = []

for u,g,d in allHours:
    allratingsPerUser[u].add(g)
    allratingsPerGame[g].add(u)
    
for u,g,d in hoursValid:
    ratingsPerUser[u].append(g)
    ratingsPerGame[g].append(u)
    if(g not in randomGame):
        randomGame.append(g)


# In[9]:


usersPerGame = defaultdict(set)
gamesPerUser = defaultdict(set)
for u,g,d in hoursTrain:
        gamesPerUser[u].add(g) 
        usersPerGame[g].add(u)


# In[10]:


def getRandom(u):
    value = randint(0, len(randomGame)-1);
    while(randomGame[value] in allratingsPerUser[u]):
        value = randint(0, len(randomGame)-1 )
    return randomGame[value] 


# In[11]:


def getNegSamp(data):
    negSamp = []
    count = 0
    for u,g,d in data:
        negSamp.append((u,g,1))
        negSamp.append((u,getRandom(u),0))
    return negSamp


# In[12]:


def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom == 0:
        return 0
    return numer / denom


# In[13]:


def mostSimilarFastItem(i, N):
    similarities = []
    users = usersPerGame[i]
    candidateItems = set()
    for u in users:
        candidateItems = candidateItems.union(gamesPerUser[u])
    for i2 in candidateItems:
        if i2 == i: continue
        sim = Jaccard(users, usersPerGame[i2])
        similarities.append((sim,i2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[14]:


def mostSimilarFastUser(u, N):
    similarities = []
    games = gamesPerUser[u]
    candidateUsers = set()
    for g in games:
        candidateUsers = candidateUsers.union(usersPerGame[g])
    for u2 in candidateUsers:
        if u2 == u: continue
        sim = Jaccard(games, gamesPerUser[u2])
        similarities.append((sim,u2))
    similarities.sort(reverse=True)
    return similarities[:N]


# In[15]:


knn_u_sim = defaultdict(set)
for u in allratingsPerUser:
    sim = mostSimilarFastUser(u, 90)
    knn_u_sim[u] = sim


# In[16]:


knn_g_sim = defaultdict(set)
for g in allratingsPerGame:
    sim = mostSimilarFastItem(g, 90)
    knn_g_sim[g] = sim


# In[17]:


knn_g = defaultdict(list)
for g in allratingsPerGame:
    for g2 in allratingsPerGame: 
        if g == g2: continue
        knn_g[g].append((Jaccard(usersPerGame[g],usersPerGame[g2]),g2))
    knn_g[g].sort(reverse= True)


# In[18]:


knn_u = defaultdict(list)
for u in allratingsPerUser:
    for u2 in allratingsPerUser: 
        if u == u2: continue
        knn_u[u].append((Jaccard(gamesPerUser[u],gamesPerUser[u2]),u2))
    knn_u[u].sort(reverse= True)


# In[19]:


neg = getNegSamp(hoursValid)
gameCount = defaultdict(int)
userCount = defaultdict(int)
totalPlayed = 0

for user,game,_ in allHours:
    gameCount[game] += 1
    totalPlayed += 1
    userCount[user] += 1 

mostPopular = [(np.log(gameCount[x]+1), x) for x in gameCount]
mostPopular.sort()
mostPopular.reverse()

return1 = set()
count = 0
for ic, i in mostPopular:
    count = ic
    return1.add(i)
    if count < 4.483: 
        break


# In[20]:


correct = 0  
played = 0
notplayed = 0
list_50 = defaultdict(list)
for u, g, i in neg:
    list_50[u].append((gameCount[g], g))
    list_50[u].sort()
    list_50[u].reverse()


# In[21]:


for u, g, i in neg:
    count_g = 0; 
    sim_G = knn_g[g][:33]
    for sim , game in sim_G:
        if u in usersPerGame[game]:
            count_g = 1 
    count = 0;
    sim_N = knn_u[u][:58]
    for sim, user in sim_N: 
        if g in gamesPerUser[user]:
            count = 1 
    for num , game in list_50[u]:
        if game == g: 
            rank = list(list_50[u]).index((num,game))
            
    if g in return1 and rank+1 <= len(list_50[u])/2:
        correct += (i ==1)
        played +=(i==1)
    elif (count == 1 and count_g == 1) and rank+1 <= len(list_50[u])/2:
        correct += (i ==1)
        played +=(i==1)
    else:
        correct += (i==0)
        notplayed += (i == 0)
accuracy = correct/len(neg)
accuracy


# In[22]:


played 


# In[23]:


notplayed


# In[24]:


list_51 = defaultdict(list)
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',') 
    list_51[u].append((gameCount[g], g))
    list_51[u].sort()
    list_51[u].reverse()


# In[25]:


predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')      
    count_g = 0; 
    sim_G = knn_g[g][:33]
    for sim , game in sim_G:
        if u in usersPerGame[game]:
            count_g = 1 
    count = 0;
    sim_N = knn_u[u][:58]
    for sim, user in sim_N: 
        if g in gamesPerUser[user]:
            count = 1 
            
    for num , game in list_51[u]:
        if game == g: 
            rank = list(list_51[u]).index((num,game))
            
    if g in return1 and rank+1 <= len(list_51[u])/2:
        predictions.write(u + ',' + g + ',' + str(1) + '\n')
    elif (count == 1 and count_g == 1) and rank+1 <= len(list_51[u])/2:
        predictions.write(u + ',' + g + ',' + str(1) + '\n')
    else:
        predictions.write(u + ',' + g + ',' + str(0) + '\n')
predictions.close()


# In[26]:


### Task 2 ####


# In[27]:


trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)


# In[28]:


def MSE(alpha, betaU, betaI):
    total = 0
    for u, g, d in hoursValid:
        if not u in betaU:
            bu = 0
        else:
            bu = betaU[u]
        if not g in betaI:
            bi = 0
        else:
            bi = betaI[g]
        our_pred = alpha + bu + bi
        total += (d['hours_transformed'] - our_pred) ** 2 
    total /= len(hoursValid)
    return total


# In[29]:


def MSE_test(alpha, betaU, betaI):
    total = 0
    for u, g, d in hoursTest:
        if not u in betaU:
            bu = 0
        else:
            bu = betaU[u]
        if not g in betaI:
            bi = 0
        else:
            bi = betaI[g]
        our_pred = alpha + bu + bi
        total += (d['hours_transformed'] - our_pred) ** 2 
    total /= len(hoursTest)
    return total


# In[30]:


alpha = globalAverage


# In[31]:


def update(alpha, userAverage, itemAverage, lam = 1):
    finish = False
    prevMSE = float('inf')
    prev_alpha = alpha
    prev_betaU = userAverage
    prev_betaI = itemAverage
    count = 0
    while not finish:
        for u, g, d in hoursTrain:
            alpha += (d['hours_transformed'] - prev_betaU[u] - 
                              prev_betaI[g])
        alpha /= len(hoursTrain)
        
        for u,g,d in hoursTrain:
            betaU[u] += (d['hours_transformed'] - alpha - prev_betaI[g])
        for u in betaU:
            betaU[u] /= (lam + num_item[u])
            
        for u, g, d in hoursTrain:
            betaI[g] += (d['hours_transformed'] - alpha - prev_betaU[u])
        for g in betaI:
            betaI[g] /= (lam + num_user[g])
            
        validMSE = MSE(alpha, betaU, betaI)
        if validMSE >= prevMSE:
            finish = True
        else:
            prev_alpha = alpha
            prev_betaU = betaU
            prev_betaI = betaI
            prevMSE = validMSE
            count += 1 
    return prev_alpha, prev_betaU, prev_betaI, count 


# In[32]:


def final_update(alpha, userAverage, itemAverage, cycles, lam = 1,):
    finish = False 
    prevMSE = float('inf')
    prev_alpha = alpha
    prev_betaU = userAverage
    prev_betaI = itemAverage
    count = 0
    while not finish:
        for u, g, d in allHours:
            alpha += (d['hours_transformed'] - prev_betaU[u] - 
                              prev_betaI[g])
        alpha /= len(allHours)
        
        for u,g,d in allHours:
            betaU[u] += (d['hours_transformed'] - alpha - prev_betaI[g])
        for u in betaU:
            betaU[u] /= (lam + num_item[u])
            
        for u, g, d in allHours:
            betaI[g] += (d['hours_transformed'] - alpha - prev_betaU[u])
        for g in betaI:
            betaI[g] /= (lam + num_user[g])
        
        if count >= cycles:
            finish = True
        else:
            prev_alpha = alpha
            prev_betaU = betaU
            prev_betaI = betaI
            prevMSE = validMSE
            count += 1 
    return prev_alpha, prev_betaU, prev_betaI


# In[33]:


alpha = globalAverage 
userAverage = defaultdict(float)
itemAverage = defaultdict(float)

betaU = defaultdict(list)
betaI = defaultdict(list)
item = defaultdict(list)
user = defaultdict(list)
num_item = defaultdict(int)
num_user = defaultdict(int)

for u, g, d in hoursTrain:
    betaU[u] = 0
    betaI[g] = 0
    user[u].append(d['hours_transformed'])
    item[g].append(d['hours_transformed'])
    num_user[g] += 1 
    num_item[u] += 1 
    
gammaU = {u: np.random.normal(scale=0.01, size=1)for u in num_item}
gammaI = {g: np.random.normal(scale=0.01, size=1)for g in num_user}


for u in user:
    userAverage[u] = sum(user[u])/len(user[u])
for g in item:
    itemAverage[g] = sum(item[g])/len(item[g])


# In[34]:


final_alpha, final_betaU, final_betaI, cycles = update(alpha, userAverage, itemAverage, lam = 6)


# In[35]:


validMSE = MSE(final_alpha, final_betaU, final_betaI)
validMSE


# In[36]:


alpha = globalAverage 
userAverage = defaultdict(float)
itemAverage = defaultdict(float)

betaU = defaultdict(list)
betaI = defaultdict(list)
item = defaultdict(list)
user = defaultdict(list)
num_item = defaultdict(int)
num_user = defaultdict(int)

for u, g, d in allHours:
    betaU[u] = 0
    betaI[g] = 0
    user[u].append(d['hours_transformed'])
    item[g].append(d['hours_transformed'])
    num_user[g] += 1 
    num_item[u] += 1 

for u in user:
    userAverage[u] = sum(user[u])/len(user[u])
for g in item:
    itemAverage[g] = sum(item[g])/len(item[g])


# In[37]:


final_alpha, final_betaU, final_betaI = final_update(alpha, userAverage, itemAverage,cycles,lam = 6)


# In[38]:


ValidMSE = MSE(final_alpha, final_betaU, final_betaI)
ValidMSE


# In[39]:


predictions = open("predictions_Hours.csv", 'w')
for l in open("pairs_Hours.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    predictions.write(u + ',' + g + ',' + str(final_alpha + final_betaU[u] + final_betaI[g]) + '\n')
predictions.close()


# In[ ]:




