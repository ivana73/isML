import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sb

data = pd.read_csv('fuel_consumption.csv')
print('Stampanje prvih 5 redova:')
print (data.head())

print('Stampanje osnovnih podataka o datasetu:')
print(data.info())

print('Prikaz statistickih atributa:')
print (data.describe())

#Brisanje redova sa NaN vrednostima
data=data.dropna(axis=0, how='any')

#Korelacija medju datim kontinualnim atributima
plt.figure() 
plt.title('Korelacija kontinualnih atributa sa emisijom CO2')
sb.heatmap(data.corr() , annot = True, fmt = '.2f' )
plt.show()

# Stampanje grafika zavisnosti CO2 emisije u zavisnosti od kontinualnih atributa 
# Promenljive kao sto su MODEL_YEAR, ENGINE_SIZE i CYLINDERS, iako se sastoje od diskretnih vrednosti, mozemo posmatrati
# kao kontinualne zato sto postoji redosled (ordering), te ove promenljive nisu klasicne kategoricke i vrednosti nisu oznake klasa
# vec imaju spektar dostupnih vrednosti.
# Takodje, promenljiva MODEL_YEAR ima jednu jedinu vrednost za svaki unos, 2014, te je ova promenljiva redudantna i nadalje
# nece biti uzimana u obzir. 
fig, ((ax1, ax2,ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,15))
fig.suptitle('Grafici CO2 emisije u zavisnosti od kontinualnih promenljivih iz dataseta')
ax1.scatter(data['ENGINESIZE'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax1.set(xlabel='ENGINESIZE', ylabel='CO2EMISSIONS')
ax2.scatter(data['CYLINDERS'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax2.set(xlabel='CYLINDERS')
ax3.scatter(data['FUELCONSUMPTION_CITY'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax3.set(xlabel='FUELCONSUMPTION_CITY')
ax4.scatter(data['FUELCONSUMPTION_HWY'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax4.set(xlabel='FUELCONSUMPTION_HWY', ylabel='CO2EMISSIONS')
ax5.scatter(data['FUELCONSUMPTION_COMB'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax5.set(xlabel='FUELCONSUMPTION_COMB')
ax6.scatter(data['FUELCONSUMPTION_COMB_MPG'], data['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax6.set(xlabel='FUELCONSUMPTION_COMB_MPG')
plt.show()


# Stampanje histograma CO2 emisije u zavisnosti od kategorickih atributa. Pored vec pomenute varijable MODEL_YEAR,
# ni kategoricka varijabla MODEL nece biti prikazana iz razloga sto ima preveliki broj klasa (659 na ukupno 1060 
# sampleova ovog dataseta) te njeno prikazivanje nema nikakvu analiticku vrednost.
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Histogrami CO2 emisije u zavisnosti od kategorickih promenljivih iz dataseta')
b1=sb.barplot( ax=ax1,x = 'MAKE' , y = 'CO2EMISSIONS' , data =data,errorbar=None)
b1.set(xticklabels=[])
b2=sb.barplot( ax=ax2,x = 'VEHICLECLASS' , y = 'CO2EMISSIONS' , data =data,errorbar=None)
b2.set(xticklabels=[]) 
b2.set(ylabel=None)
b3=sb.barplot( ax=ax3,x = 'TRANSMISSION' , y = 'CO2EMISSIONS' , data =data,errorbar=None)
b3.set(xticklabels=[]) 
b4=sb.barplot( ax=ax4,x = 'FUELTYPE' , y = 'CO2EMISSIONS' , data =data,errorbar=None)
b4.set(xticklabels=[]) 
b4.set(ylabel=None)
plt.show()

# Nakon prikaza grafika i histograma zavisnosti emisije CO2 od svih atributa, potrebno je odabrati podskup
# atributa koji ce biti korisceni za trening. Sa tabele korelacije kontinualnih atributa, jasno je da sve promenljive imaju visoku
# korelaciju sa izlaznim nivoom emisije CO2 i da takodje imaju visok stepen medjusobne korelacije. Ocekivano, medju njima
# FUELCONSUMPTION_CITY,FUELCONSUMPTION_HWY, FUELCONSUMPTION_COMB i FUELCONSUMPTION_COMB_MPG imaju izrazito visok koeficijent
# korelacije tj. antikorelacije u slucaju FUELCONSUMPTION_COMB_MPG, te je dovoljno odabrati jednu
# od ovih promenljivih za trening i u ovom slucaju bira se FUELCONSUMPTION_COMB jer ima najveci stepen korelacije sa CO2EMISSIONS.
# Pored nje, od kontinualnih promenljivih bira se jos i ENGINESIZE i CYLINDERS, jer mogu da obezbede dodatne informacije, pored FUELCONSUMPTION_COMB.
# Takodje, logicki gledano, ovaj podskup atributa ima najvise smisla iskoristiti u ovom problemu modelovanja.

# U slucaj u kategorickih promenljivih, histogrami su prikazani. Ove promenljive predstavljaju razlicite klase u okviru jednog
# atributa i iz tog razloga su veoma nepovoljne za plotovanje i problem regresije, jer ne postoji nacin da se medju njima
# uvede nekakav redosled. Nacin da se one enkodiraju i iskoriste u ovom slucaju je da se transformisu u one-hot encoding
# s tim sto je potrebno uvesti broj promenljivih jednak broju razlicitih klasa. Prebrojavanjem broja razlicitih klasa utvrdjeno je sledece:
print("Broj unikatnih klasa u okviru kategorickih atributa")
print("N(FUELTYPE):", data['FUELTYPE'].nunique())
print("N(TRANSMISSION):",data['TRANSMISSION'].nunique())
print("N(VEHICLECLASS):",data['VEHICLECLASS'].nunique())
print("N(MODEL):",data['MODEL'].nunique())
print("N(MAKE):",data['MAKE'].nunique())
# Kao sto je vec opisano, kako bi se kategoricke varijable uvele u model potreban je broj novih atributa jednak broju klasa
# te je za sve slucajeve osim za FUELTYPE taj broj prevelik i komputaciono zahtevan. Takodje, od svih navedenih parametara, jedino
# FUELTYPE ima smisla koristiti u ovom slucaju jer predstavlja parametar koji moze da utice na emisiju CO2 iz autamobila

# Zavisnost emisije CO2 u odnosu na potrosnju goriva, predstavljeno za razlicite klase goriva.
data1=data[data['FUELTYPE']=='E']
data2=data[data['FUELTYPE']=='Z']
data3=data[data['FUELTYPE']=='D']
data4=data[data['FUELTYPE']=='X']
plt.figure()
plt.title('Grafik zavisnosti CO2 emisije od potrosnje goriva za razlicite tipove goriva')
plt.scatter(data1['FUELCONSUMPTION_COMB'], data1['CO2EMISSIONS'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
plt.scatter(data2['FUELCONSUMPTION_COMB'], data2['CO2EMISSIONS'], s=23, c='green', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
plt.scatter(data3['FUELCONSUMPTION_COMB'], data3['CO2EMISSIONS'], s=23, c='blue', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
plt.scatter(data4['FUELCONSUMPTION_COMB'], data4['CO2EMISSIONS'], s=23, c='black', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.legend(['Tip E', 'Tip Z', 'Tip D', 'Tip X'])
plt.show()
# Dodatno, analizom grafika kontinualne promenljive, izdvaja se vise od jedne linearne zavisnosti na grafiku. Prikazom prethodnog grafika,
# na kome je taj originalni grafik podeljen zavisno od tipa goriva, vidi se da se linearne zavisnost menja upravo zavisno od goriva koje se trosi. Unosom varijable
# FUELTYPE u nas model moguce je da ce se ova zavisnost uspesno 'uhvatiti' predlozenim modelom, racunajuci na dodatnu informaciju koja ova promenljiva nosi.

# Konacno, skup atributa koji ce nadalje biti koriscen je sledeci : FUELCONSUMPTION_COMB,FUELTYPE,ENGINESIZE i CYLINDERS
# Sada je potrebno idvojiti kolone odredjene za treniranje u dve odvojene matrice:
X=data[['FUELCONSUMPTION_COMB','FUELTYPE','ENGINESIZE','CYLINDERS']]
y=data['CO2EMISSIONS']
# Sada je potrebno izvrsiti transformacije kategoricke promenljive FUELTYPE i pripremiti ovu i ostale promenljive za trening
fuel_onehot = pd.get_dummies(X['FUELTYPE'], prefix='FUELTYPE')
X.drop(columns=['FUELTYPE'], inplace=True)
X = X.join(pd.DataFrame(data=fuel_onehot))
# Konacno, potrebno je skalirati sve kolone na vrednosti izmedju 0 i 1, radi lakseg treniranja i podeliti podatke na trening i test skup
X=(X-X.min())/(X.max()-X.min())
y=(y-y.min())/(y.max()-y.min())
X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=0.9,random_state=234,shuffle=True)

# Definisanje klase za kreiranje naseg modela
class LinearRegressionGradientDescent:
    def __init__(self):
        self.coeff=None
        self.features=None
        self.target=None
        self.mse_history=None

    def set_coefficients(self,*args):
        self.coeff = np.array(args).reshape(-1,1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s=pow(predicted - self.target,2).sum()
        return (0.5/len(self.features))*s

    def predict(self,features):
        features=features.copy(deep=True)
        features.insert(0,'c0',np.ones((len(features),1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1,1).flatten()

    def gradient_descent_step(self,learning_rate):
        predicted=self.features.dot(self.coeff)
        s=self.features.T.dot(predicted-self.target)
        gradient=(1./len(self.features))*s
        self.coeff=self.coeff-learning_rate*gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self,learning_rate, num_iterations=100):
        self.mse_history=[]
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
            self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history
    
    def fit(self,features,target):
        self.features=features.copy(deep=True)
        coeff_shape=len(features.columns)+1
        self.coeff=np.zeros(shape=coeff_shape).reshape(-1,1)
        self.features.insert(0,'c0',np.ones((len(features),1)))
        self.features=self.features.to_numpy()
        self.target=target.to_numpy().reshape(-1,1)

# Trening sa predefinisanim modelom
lr_model=LinearRegression()
lr_model.fit(X_train,y_train)
c=np.concatenate((np.array([lr_model.intercept_]),lr_model.coef_))
print("Parametri modela treniranog sklearn funkcijom:",c)

# Trening sa lokalnom implementacijom modela
lrgd=LinearRegressionGradientDescent()
lrgd.fit(X_train,y_train)
learning_rate=1.0
res_coeff,mse_history=lrgd.perform_gradient_descent(learning_rate,20)
print("Parametri modela treniranog lokalnom implementacijom :",res_coeff.flatten())

# MSE greska na izdvojenom test setu - Evaluacija modela
lrgd.fit(X_test,y_test)

lrgd.set_coefficients(c)
print("MSE greska modela- sklearn funkcija:",lrgd.cost())

lrgd.set_coefficients(res_coeff)
print("MSE greska modela- lokalna implementacija:",lrgd.cost())


# Crtanje grafika MSE greske na trening skupu u toku treninga lokalno implementiranog modela, kroz iteracije. Ocekivano, MSE greska opada, najznacajnije u prvih par iteracija 
plt.figure('MSE greska u toku treninga')
plt.plot(np.arange(0,len(mse_history),1),mse_history)
plt.xlabel('Broj iteracije',fontsize=13)
plt.ylabel('vrednost MSE', fontsize=13)
plt.xticks(np.arange(0,len(mse_history),2))
plt.title('MSE greska u toku treniranja')
plt.tight_layout()
plt.show()

# Oba predlozena modela su uspesno istrenirana, sa tom razlikom gde sklearn model ipak ima znacajno manju gresku na test skupu. Glavni razlog
# tome svakako jesu bolji algoritam optimizacije i izbora hiperparametera, koji se u slucaju sklearna prilagodjava gradijentu za svaki pojedinaci parametar
# te je otuda taj model omogucen da postize znacajno nize vrednosti MSE greske. I pored toga, oba modela veoma dobro vrse zadatak linearne regresije i predvidjanja
# vrednosti emisije CO2
