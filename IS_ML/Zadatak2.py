import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
data = pd.read_csv('cakes.csv')
print('Stampanje prvih 5 redova:')
print (data.head())

print('Stampanje osnovnih podataka o datasetu:')
print(data.info())

print('Prikaz statistickih atributa:')
print (data.describe())

#Brisanje redova sa NaN vrednostima
data=data.dropna(axis=0, how='any')

# Transformacija kategoricke binarne promenljive za tip kolaca u numericku, kako bi mogli uociti
# korelaciju ostalih atributa sa njom
data['type'].replace(['cupcake', 'muffin'],
                        [0, 1], inplace=True)

#Korelacija medju datim kontinualnim atributima
# Sve promenljive date u datasetu imaju numericke vrednosti i prirodni poredak (cak i ako su kategoricke 
# kao npr eggs), tako se mogu posmatrati u sirem smislu kao kontinualne i prikazati zajedno sa ostalima
plt.figure() 
plt.title('Korelacija kontinualnih atributa sa tipom kolaca')
sb.heatmap(data.corr() , annot = True, fmt = '.2f' )
plt.show()



# Stampanje grafika tipa kolaca u zavisnosti od kontinualnih atributa - histogrami 
fig, ((ax1, ax2,ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,15))
fig.suptitle('Histogrami tipa kolaca u zavisnosti od kontinualnih promenljivih iz dataseta')
sb.histplot(data,x='flour',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax1)
ax1.set_xlabel('flour')
ax1.legend(['muffin','cupcake'],loc='upper right')
sb.histplot(data,x='eggs',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax2)
ax2.set_xlabel('eggs')
ax2.legend(['muffin','cupcake'],loc='upper right')
sb.histplot(data,x='sugar',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax3)
ax3.set_xlabel('sugar')
ax3.legend(['muffin','cupcake'],loc='upper right')
sb.histplot(data,x='milk',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax4)
ax4.set_xlabel('milk')
ax4.legend(['muffin','cupcake'],loc='upper right')
sb.histplot(data,x='butter',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax5)
ax5.set_xlabel('butter')
ax5.legend(['muffin','cupcake'],loc='upper right')
sb.histplot(data,x='baking_powder',hue='type',hue_order=[0,1],multiple='fill',legend=False,ax=ax6)
ax6.set_xlabel('baking_powder')
ax6.legend(['muffin','cupcake'],loc='upper right')
plt.show()

# Stampanje grafika tipa kolaca u zavisnosti od kontinualnih atributa - tacke u Dekartovom koordinatnom sistemu
fig, ((ax1, ax2,ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15,15))
fig.suptitle('Grafici tipa kolaca u zavisnosti od kontinualnih promenljivih iz dataseta')
ax1.scatter(data['flour'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax1.set(xlabel='flour',ylabel='type')
ax2.scatter(data['eggs'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax2.set(xlabel='eggs')
ax3.scatter(data['sugar'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax3.set(xlabel='sugar')
ax4.scatter(data['milk'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax4.set(xlabel='milk',ylabel='type')
ax5.scatter(data['butter'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax5.set(xlabel='butter')
ax6.scatter(data['baking_powder'], data['type'], s=23, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=2)
ax6.set(xlabel='baking_powder')
plt.show()


# Konacno, potrebno je odabrati atribute koji ce biti korisceni za predikciju u nasem modelu. U ovom konkretnom slucaju,
# Broj primera je jako mali (oko 100), broj atributa takodje (6), i nijedan nema dovoljno veliku korelaciju sa ostalima da bi 
# sa sigurnoscu mogli da ga izostavimo. Iz tog razloga, svi atributi dati ovde ce biti korisceni u predikciji, jer nece dovoditi do 
# prevelikih komputacionih problema

# Sada je potrebno idvojiti kolone odredjene za treniranje u dve odvojene matrice:
X=data[['flour','eggs','sugar','milk','butter','baking_powder']]
y=data['type']
# Konacno, potrebno je skalirati sve kolone na vrednosti izmedju 0 i 1, radi lakseg treniranja i podeliti podatke na trening i test skup
X=(X-X.min())/(X.max()-X.min())
y=(y-y.min())/(y.max()-y.min())
X_train,X_test,y_train,y_test=train_test_split(X,y, train_size=0.8,random_state=234,shuffle=True)

# Predikcija sa ugradjenim KNN modelom
model = KNeighborsClassifier(n_neighbors=20)
model.fit(X_train,y_train)
y_pred_sklearn=model.predict(X_test)

# Implementacija lokalne KNN funkcije i klase
class KNNClassifier:
    def __init__(self, num_neigbors=20):
        self.features=None
        self.targets=None
        self.num_neighbors=num_neigbors

    def fit(self,features,labels):
        self.features=features.to_numpy()
        self.labels=labels.to_numpy()

    def predict(self,test_features):
        test_features=test_features.to_numpy()
        predictions=list()
        for i in range(len(test_features)):
            query=test_features[i,:]
            predictions+=[self.predict_one(query)]
        return np.array(predictions)

    def predict_one(self, query):
        diffs=np.linalg.norm(self.features-query,axis=1)
        ind=np.argsort(diffs)
        labels=self.labels[ind]
        return 1 if np.sum(labels[:self.num_neighbors])>self.num_neighbors//2 else 0

# Instanciranje klase KNN klasifikatora i predikcija na test setu
model=KNNClassifier()
model.fit(X_train,y_train)
y_pred_local=model.predict(X_test)

# Prikaz konfuzione matrice za dati test set, koriscene predikcije iz sklearn funkcije
cm=metrics.confusion_matrix(y_test,y_pred_sklearn)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['cupcake', 'muffin'])
cm_display.plot()
plt.title('Konfuziona matrica tipa kolaca - sklearn ugradjena funkcija')
plt.show()

accuracy=(cm[0][0]+cm[1][1])/len(y_test)
print("Procenat tacnih klasifikacija sklearn funkcije: {0}%".format(100*accuracy))

# Prikaz konfuzione matrice za dati test set, koriscene predikcije iz lokalne funkcije
cm=metrics.confusion_matrix(y_test,y_pred_local)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['cupcake', 'muffin'])
cm_display.plot()
plt.title('Konfuziona matrica tipa kolaca - lokalna KNN funkcija')
plt.show()

accuracy=(cm[0][0]+cm[1][1])/len(y_test)
print("Procenat tacnih klasifikacija lokalne KNN funkcije: {0}%".format(100*accuracy))




