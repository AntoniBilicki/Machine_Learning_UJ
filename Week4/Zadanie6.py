

# Zadanie6: Rozważ dane zawierające nagłówki, przy czym są to fake newsy (1298) oraz prawdziwe tytuły (1968).
# Jeden nagłówek to jeden rekord.
# Wyznacz częstości występowania wszystkich słów ze wszystkich nagłówków.
# Jakie słowa (oprócz stopwordsów) najczęściej pojawiały się w realnych a jakie w fałszywych nagłówkach?
# Będziemy tworzyć klasyfiaktor dla tych danych w oparciu o regresję logistyczną oraz drzewa losowe/lasy losowe.
# Każdy nagłówek będzie reprezentowany w postaci wektora zer i jedynek w zależności od występowania danego słowa
# (długość wektora = liczba wszystkich unikatowych słów, może warto jednak zawęzić? albo potraktować jako hiperparametr).
# Podziel dane na 3 grupy: 70% zbiór treningowy, 15% zbiór walidacyjny, 15% zbiór testowy.
# Przetestuj różne zestawy hiperparametrów na zbiorze walidacyjnym. Skomentuj otrzymane wyniki.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# loading the data
with open('real.txt') as f:
    real_data = f.readlines()
f.close()

with open('fake.txt') as f:
    fake_data = f.readlines()
f.close()


# flatten the data for further processing:
def flatten(l):
    flat_list = []
    for sublist in l:
        flat_list.extend(sublist.split())
    return flat_list


real_flat_data = flatten(real_data)
fake_flat_data = flatten(fake_data)

# count occurrences
real_set, real_count = np.unique(real_flat_data, return_counts=True)
fake_set, fake_count = np.unique(fake_flat_data, return_counts=True)

# sort by occurrences
real_count, real_set = zip(*sorted(zip(real_count, real_set), reverse=True))
fake_count, fake_set = zip(*sorted(zip(fake_count, fake_set), reverse=True))

#plot data
plt.subplot(2, 1, 1)
plt.bar(real_set[:20], real_count[:20])
plt.title('Top 20 word occurrences in real news headlines')
plt.subplot(2, 1, 2)
plt.bar(fake_set[:20], fake_count[:20])
plt.title('Top 20 word occurrences in fake news headlines')
plt.show()

# Converting the data into 0,1 vector
total_set = np.unique(np.append(list(real_set), fake_set))


def transf_data_2_bool(data, key):
    output_data = []
    for idx, line in enumerate(data):
        output_data.append(np.isin(key, line))
    return output_data


# defining x and y
bool_total_data = transf_data_2_bool([line.split() for line in real_data + fake_data], total_set)
y = np.concatenate((np.ones(len(real_data), dtype=int), np.zeros(len(fake_data), dtype=int)))

# Splitting the data into train, test, val

X_train, X_temp, y_train, y_temp = train_test_split(bool_total_data, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.3)

# Create model
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Omowienie wynikow: Model jest zdolny do przewidywania "fake newsow" na podstawie pojedynczych slow ze zdolnoscia
# okolo 80%. Warto jednak zaznaczyc, ze moze to w duzej mierze wynikac z zastosowanego zestawu 'fake newsow'. W celu
# zwiekszenia realnej wartosci modelu zalecana bylaby lemmatyzacja slow, oraz przeprowadzenie grupowania po 2 - 3 wyrazy
# powinno to zwiekszyc zdolnosc modelu do rozpoznawania "realnych" polaczen slow od przypadkowych ulozen.