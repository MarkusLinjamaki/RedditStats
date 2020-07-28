# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:53:56 2020


@author: Markus
"""

#%%
import math
import pandas
import numpy
import statistics
import matplotlib.pyplot
import scipy
import scipy.stats
import sklearn
import sklearn.linear_model
import pandas
from sklearn.metrics import mean_squared_error
import statsmodels.api
import pylab

myDataFrame = pandas.read_csv('harjoitustyodata.csv')
myDataArray = myDataFrame.to_numpy()
myDataMatrix = numpy.matrix(myDataArray)

katTaulu = {1:'alt.atheism', 2:'comp.graphicssta', 3:'comp.os.ms-windows.misc', 4:'comp.sys.ibm.pc.hardware',
5 :'comp.sys.mac.hardware', 6:'comp.windows.x', 7:'misc.forsale', 8:'rec.autos', 9:'rec.motorcycles',
10:'rec.sport.baseball', 11:'rec.sport.hockey', 12:'sci.crypt', 13:'sci.electronicsk', 14:'sci.med', 15:'sci.space',
16:'soc.religion.christian', 17:'talk.politics.guns', 18:'talk.politics.mideast', 19:'talk.politics.misc', 20:'talk.religion.misc'}



#%%
### plottaa histogramin saadusta datasta
def histogrammi(dataArray,otsikko, valiotsikko):
    myfigure, myaxes = matplotlib.pyplot.subplots();
    myaxes.hist(dataArray, bins=20, histtype='bar');
    matplotlib.pyplot.title(otsikko)
    if len(valiotsikko) > 0:       
        matplotlib.pyplot.xlabel(valiotsikko)
            
    
#%% 1 kohta tietyn uutisryhmän tietyn sanan esiintymien keskiaro, mediaani, hajonta, kvantiilit[0.01,0.99] 
## sanojen yhteisesiintymämäärä
def sananTunnusluvut(aineisto,groupID,sana):
    data = aineisto.loc[aineisto.loc[:,'groupID'] == groupID,sana]
    dataArray = data.to_numpy()
    tLuvut = tunnusluvut(data)
    histogrammi(dataArray,katTaulu[groupID],'sana:' + sana)
    summa = 0
    for i in dataArray:
        summa = summa + int(i)
    tLuvut['sanojenLKM'] = summa
    return tLuvut

#%%
### laskee annetusta datasta tunnusluvut ja palauttaa ne sanakirjassa
def tunnusluvut(data):
    ka = round(statistics.mean(data),3)
    med = round(statistics.median(data),2)
    hajonta = round(statistics.stdev(data),2)
    kvantiilit = numpy.quantile(data,[0.25,0.75])
    tLuvut = {'keskiarvo':ka, 'mediaani':med, 'hajonta':hajonta, 'kvantiilit':kvantiilit}
    return tLuvut
#%%
### Muodostaa piirakkagraafin annetusta datasta
def piirakkaGraafi(sektorinimet,sanat, sana):
    myfigure, myaxes = matplotlib.pyplot.subplots()
    nimet = []
    for i in sektorinimet:
        nimet.append(katTaulu[i])
    summa = 0
    osuudet = []
    for i in sanat:
        summa = summa + int(i)
    for i in sanat:
        osuus = int(i) / summa
        osuudet.append(osuus)
    myaxes.pie(osuudet,labels=nimet,autopct='%.1f');
    matplotlib.pyplot.xlabel('sana:' + sana)
    


#%%
### Palauttaa listan tietyn uutisryhmän sanojen pituudesta
def sanojenPituudet(dataFrame, groupID):
    lista = []
    data = dataFrame.loc[dataFrame.loc[:,'groupID'] == groupID,:]
    dataArray = data.to_numpy()
    for i in range(len(dataArray)):
        summa = 0
        for a in range(5,5 + 962 + 1):
            summa = summa + int(myDataArray[i][a])
        lista.append(summa)
    return lista



        
#%% 
def tietojenTulostus(groupID, tiedot):
    print(katTaulu[groupID], 'tunnusluvut ', end = '')
    for k,v in tiedot.items():
        print(k,': ',v,' ' ,sep = '' ,end = '')
    print('')


#%%
### Laskee sanoje esiintymät koko aineistosta
def sanojenEsiintymat(sana):
    kaikkiSanat = []
    ## jokaisen uutisryhmän sanojen määrä
    for i in range(1,21):
        summa = 0
        # tietyn uutisryhmän sanat
        ryhma = katTaulu[i]
        data = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == i,sana]
        for a in data:
            summa = summa + a
        kaikkiSanat.append(summa)
        
    return kaikkiSanat


#%%
### Laskee korrelaation kahden sana välillä
def korrelaatio(sana1, sana2):
    tiedot1 = sanojenEsiintymat(sana1)
    tiedot2 = sanojenEsiintymat(sana2)
    matriisi = numpy.matrix([tiedot1,tiedot2])
    korrelaatio = numpy.corrcoef(matriisi)
    print(korrelaatio)
    
#%%
### pylväsdiagrammi annetusta datasta
def barplot(data,otsikko):
    import numpy as np
    import matplotlib.pyplot as plt
    nimet = []
    height = []
    for i in range(len(data)):
        nimet.append(data[i][0])
        height.append(abs(data[i][1]))
    bars = (*nimet,)
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.show()
   
    
    
#%% 
### Laskee sentimentaali arvo kaikille uutisryhmille, järjestää ne keskiarvon ja hajonnan mukaan
def sentimentaali():
    tiedot = []
    for i in range(1,21):
        tunnusluvut = sananTunnusluvut(myDataFrame, i, 'meanvalences')
        ryhmanTiedot = [katTaulu[i],tunnusluvut['keskiarvo'], tunnusluvut['mediaani'], tunnusluvut['hajonta'], tunnusluvut['kvantiilit']]
        tiedot.append(ryhmanTiedot)
    ## lajitellaan lambda lausetta käyttäen keskiarvon,mediaanin, käänteisen hajonnan ja nimen mukaan
    tiedot.sort(key = lambda nimi : nimi[0])
    tiedot.sort(key = lambda hajonta : hajonta[3], reverse = True)
    tiedot.sort(key = lambda med : med[2])
    tiedot.sort(key = lambda ka : ka[1])

#%% sana 'freedom' uutisryhmissä sci.crypt, talk.politics.guns talk.politics.mideast ja talk.politics.misc.
tiedot1 = sananTunnusluvut(myDataFrame, 12, 'freedom')
tiedot2 = sananTunnusluvut(myDataFrame, 17, 'freedom')
tiedot3 = sananTunnusluvut(myDataFrame, 18, 'freedom')
tiedot4 = sananTunnusluvut(myDataFrame, 19, 'freedom')
print(' SANA FREEDOM', end = '\n')
tietojenTulostus(12, tiedot1)
tietojenTulostus(17, tiedot2)
tietojenTulostus(18, tiedot3)
tietojenTulostus(19,tiedot4)
### Vertaillaan sanamääriä
sektorinimet = [12,17,18,19]
kaikkiSanat = [tiedot1['sanojenLKM'], tiedot2['sanojenLKM'] ,tiedot3['sanojenLKM'] ,tiedot4['sanojenLKM']]
piirakkaGraafi(sektorinimet, kaikkiSanat,'freedom')

### sana 'nation' utisryhmissä talk.politics.guns(17), talk.politics.mideast(18) ja talk.politics.misc.(19)
print(' SANA nation', end = '\n')
tiedot1 = sananTunnusluvut(myDataFrame, 17, 'nation')
tiedot2 = sananTunnusluvut(myDataFrame, 18, 'nation')
tiedot3 = sananTunnusluvut(myDataFrame, 19, 'nation')

tietojenTulostus(17, tiedot1)
tietojenTulostus(18, tiedot2)
tietojenTulostus(19,tiedot3)

sektorinimet = [17,18,19]
kaikkiSanat = [tiedot1['sanojenLKM'], tiedot2['sanojenLKM'] ,tiedot3['sanojenLKM']]
piirakkaGraafi(sektorinimet, kaikkiSanat,'nation')

### sana 'logic' uutisryhmissä alt.atheism(1), sci.electronics(13), talk.politics.misc(19( ja talk.religion.misc.(20))
print(' SANA logic', end = '\n')
tiedot1 = sananTunnusluvut(myDataFrame, 1, 'logic')
tiedot2 = sananTunnusluvut(myDataFrame, 13, 'logic')
tiedot3 = sananTunnusluvut(myDataFrame, 19, 'logic')
tiedot4 = sananTunnusluvut(myDataFrame, 20, 'logic')

tietojenTulostus(1, tiedot1)
tietojenTulostus(13, tiedot2)
tietojenTulostus(19, tiedot3)
tietojenTulostus(20,tiedot4)

sektorinimet = [1,13,19,20]
kaikkiSanat = [tiedot1['sanojenLKM'], tiedot2['sanojenLKM'] ,tiedot3['sanojenLKM'] ,tiedot4['sanojenLKM']]
piirakkaGraafi(sektorinimet, kaikkiSanat,'logic')

### sana 'normal' uutisryhmissä comp.graphics(2), comp.windows.x(6), sci.electronics(13) ja sci.med.(14)

print(' SANA normal', end = '\n')
tiedot1 = sananTunnusluvut(myDataFrame, 2, 'normal')
tiedot2 = sananTunnusluvut(myDataFrame, 6, 'normal')
tiedot3 = sananTunnusluvut(myDataFrame, 13, 'normal')
tiedot4 = sananTunnusluvut(myDataFrame, 14, 'normal')

tietojenTulostus(2, tiedot1)
tietojenTulostus(6, tiedot2)
tietojenTulostus(13, tiedot3)
tietojenTulostus(14,tiedot4)

sektorinimet = [2,6,13,14]
kaikkiSanat = [tiedot1['sanojenLKM'], tiedot2['sanojenLKM'] ,tiedot3['sanojenLKM'] ,tiedot4['sanojenLKM']]
piirakkaGraafi(sektorinimet, kaikkiSanat,'normal')

### sana 'program' uutisryhmissä comp.graphics, comp.windows.x, talk.politics.misc ja comp.sys.mac.hardware.

#%%
### Viestien pituudet uutisryhmissä rec.sport.baseball(10) ja rec.sport.hockey.(11)
tiedot1 = sanojenPituudet(myDataFrame, 10)
histogrammi(tiedot1, katTaulu[10], 'sanojen pituudet')
logTiedot1 = [math.log(a) for a in tiedot1]
histogrammi(logTiedot1, katTaulu[10], 'sanojen pituudet (logaritmi)')
tiedot2 = sanojenPituudet(myDataFrame, 11)
histogrammi(tiedot2, katTaulu[11], 'sanojen pituudet')
logTiedot2 = [math.log(a) for a in tiedot2]
histogrammi(logTiedot1, katTaulu[11], 'sanojen pituudet (logaritmi)')
t_testi1 = scipy.stats.ttest_ind(logTiedot1,logTiedot2,axis=0)
print('keskiarvo ryhmä 1:',statistics.mean(logTiedot1),'keskiarvo rthmä 2:',statistics.mean(logTiedot2))
print(t_testi1)

### Viestien pituudet uutisryhmissä rec.autos(8)   ja   rec.motorcycles(9)
tiedot3 = sanojenPituudet(myDataFrame, 8)
histogrammi(tiedot3, katTaulu[8], 'sanojen pituudet')
logTiedot3 = [math.log(a) for a in tiedot3]
histogrammi(logTiedot3, katTaulu[8], 'snojen pituudet (logaritmi)')

tiedot4 = sanojenPituudet(myDataFrame, 9)
histogrammi(tiedot4, katTaulu[9], 'sanojen pituudet')
logTiedot4 = [math.log(a) for a in tiedot4]
histogrammi(logTiedot4, katTaulu[9], 'sanojen pituudet (logaritmi)')
t_testi2 = scipy.stats.ttest_ind(logTiedot3,logTiedot4,axis=0)
print('keskiarvo ryhmä 1:',statistics.mean(logTiedot3),'keskiarvo rthmä 2:',statistics.mean(logTiedot4))
print(t_testi2)


#%% 




### Vertaillaan 'secsfrom8am' ja 'secsfrommidnight' muuttujien jakautuneisuutta

data1 = myDataFrame.loc[:,'secsfrommidnight']
histogrammi(data1, 'secsfrommidnight', "")
data2 = myDataFrame.loc[:,'secsfrom8am']
histogrammi(data2, 'secsfrom8am', "")
tiedot = tunnusluvut(data2)
print(tiedot)

### Vertaillaan viestien muunnettuja kirjoitusaikoja ryhmissä comp.graphics(2) ja soc.religion.christian(16)
data3 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2,'secsfrom8am']
data4 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 16,'secsfrom8am']
histogrammi(data1, 'secsfrom8am', katTaulu[2])
histogrammi(data2, 'secsfrom8am', katTaulu[16])
tiedot3 = tunnusluvut(data3)
tiedot4 = tunnusluvut(data4)
print(tiedot3,tiedot4)
### t-testi
t_testi = scipy.stats.ttest_ind(data3,data4,axis=0)
print(t_testi)

         
    
#%%
### Korrelaatio sanan 'jpeg' esiintymämäärän ja sanan 'gif' esiintymämäärän välillä ylikaikkien uutisryhmien viestien.
korrelaatio('jpeg', 'gif')

##Korrelaatio sanan 'write' esiintymämäärän ja sanan 'sale' esiintymämäärän välillä ylikaikkien uutisryhmien viestien.
korrelaatio('write', 'sale')

## korrelaatio sanan 'jpeg' esiintymämäärän ja sanan 'gif' esiintymämäärän välillä yliuutisryhmän comp.graphics(2)   viestien.
data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2,'jpeg']
comp_jpeg = [a for a in data1]
data2 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2,'gif']
comp_gif = [ e for e in data2]
matriisi = numpy.matrix([data1,data2])
tulos = numpy.corrcoef(matriisi)
print(tulos)



#%% Sentimentin analysointi.
### normaalisuudentesti, onko sentimenttiarvo normaalijakautunut yli koko aineiston.
data = myDataFrame.loc[:,'meanvalences']
data_array= data.to_numpy()
data_matrix = numpy.transpose(numpy.matrix(data_array))
histogrammi(data, 'meanvalences', 'normaalisuus')

#scipy.stats.probplot(data, dist = "norm", plot = pylab)
pylab.show()

#teststatistic,pvalue1=scipy.stats.shapiro(data_matrix)
#teststatistic,pvalue2=scipy.stats.normaltest(data_array,axis=0)


### jakauma uutisryhmiencomp.sys.ibm.pc.hardware(5) ja comp.sys.mac.hardware(6)   välillä
data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 5,'meanvalences']
data2 =  myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 6,'meanvalences']
t_testi1 = scipy.stats.ttest_ind(data1,data2,axis=0)
### rec.sport.baseball(10) ja rec.sport.hockey(11)
data3 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 10,'meanvalences']
data4 =  myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 11,'meanvalences']
t_testi2 = scipy.stats.ttest_ind(data3,data4,axis=0)

###rec.autos(8) ja rec.motorcyclesvälillä(9)?
data5 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 8,'meanvalences']
data6 =  myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 9,'meanvalences']
t_testi3 = scipy.stats.ttest_ind(data5,data6,axis=0)

print(t_testi1)
print(t_testi2)
print(t_testi3)

#%%
### yritetään  ennustaa  viestin sanamäärien perusteella, mihinryhmään viesti kuuluu.
### Verrataan   kahta   uutisryhmää: comp.graphics(2) ja sci.space(15) ja niidendokumentteja.
### Merkitään kullekin dokumentille tavoitemuuttujan arvoksi 1 jos se kuuluu edelliseenuutisryhmään ja -1 jos se kuuluu jälkimmäiseen.
x = []
y = []
data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2 ,'jpeg']
for a in data1:
    y.append(1)
    x.append(a)
data2 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 15 ,'jpeg']
for e in data2:
    y.append(-1)
    x.append(e)
matplotlib.pyplot.scatter(x,y)
y = numpy.transpose(numpy.matrix(y))
x = numpy.transpose(numpy.matrix(x))

tulos=sklearn.linear_model.LinearRegression().fit(x,y);
ypred=tulos.predict(x)

#y =numpy.squeeze(numpy.array(ypred))
#matplotlib.pyplot.scatter(ypred,y)

MSE1 = numpy.square(numpy.subtract(y, ypred)).mean()

#%%
x = []
y = []

data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2 ,['jpeg','earth']]
data_array1 = data1.to_numpy()
for a in data_array1:
    y.append(1)
    x.append(a)
    
data2 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 15 ,['jpeg','earth']]
data_array2 = data2.to_numpy()
for a in data_array2:
    y.append(-1)
    x.append(a)

y = numpy.transpose(numpy.matrix(y))
x = numpy.matrix(x)

tulos=sklearn.linear_model.LinearRegression().fit(x,y);
ypred=tulos.predict(x)

MSE2 = numpy.square(numpy.subtract(y, ypred)).mean()




#%%
### Lineaari regressio

x = []
y = []

data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2 ,['group1','group2','group3','group4','group5','group6','group7','group8']]
data_array1 = data1.to_numpy()
for a in data_array1:
    y.append(1)
    x.append(a)
    
data2 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 15 ,['group1','group2','group3','group4','group5','group6','group7','group8']]
data_array2 = data2.to_numpy()
for a in data_array2:
    y.append(-1)
    x.append(a)
y_matrix = numpy.transpose(numpy.matrix(y))
x_matrix = numpy.matrix(x)

tulos=sklearn.linear_model.LinearRegression().fit(x,y);
ypred=tulos.predict(x)

MSE3 = numpy.square(numpy.subtract(y, ypred)).mean()
#%%
### Lineaari regressio

x = []
y = []

data1 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 2 ,['group1','group2','group3','group4','group5','group6','group7','group8']]
data_array1 = data1.to_numpy()
for a in data_array1:
    y.append(1)
    x.append(a)
    
data2 = myDataFrame.loc[myDataFrame.loc[:,'groupID'] == 15 ,['group1','group2','group3','group4','group5','group6','group7','group8']]
data_array2 = data2.to_numpy()
for a in data_array2:
    y.append(-1)
    x.append(a)
y_matrix = numpy.transpose(numpy.matrix(y))
x_matrix = numpy.matrix(x)

tulos=sklearn.linear_model.LinearRegression().fit(x_matrix,y_matrix);
ypred=tulos.predict(x_matrix)
matplotlib.pyplot.scatter(ypred,y)

MSE3 = numpy.square(numpy.subtract(y, ypred)).mean()



ypred=numpy.squeeze(numpy.array(ypred))
onnistui = 0
for i in range(len(ypred)):
    if (ypred[i] > 0 and y[i] == 1):
        onnistui = onnistui + 1
    elif (ypred[i] < 0 and y[i] == -1):
        onnistui = onnistui + 1
    else:
        pass




