import pandas as pd
df = pd.read_csv("C:\\Users\\gezer\\Untitled Folder 2\\heart_disease_data.csv")
df.head() #veri setinin il beş satırı

df.tail() # versetinin son beş satırı

df.shape #verisetimin boyutu 

df.info #veri seti hakkında bilgi
df.isnull().sum() #eksik değer toplamları
df.isnull().any() #eksik değer var mı

df.describe() # istatistiksel değerler , standart sapma , ortalama , çeyrekler değerleri gibi

df['target'].value_counts() # target değişkeninin dağılımı

#aritmetik ortalamayı hesapla
df[['age', 'cp', 'chol', 'fbs', 'thalach']].mean()

#ortalaamdan büyük olan target değerleri
df[df['target'] > df['target'].mean()]['target'].count()

#ortalamadan küçük olan target değerleri
df[df['target'] < df['target'].mean()]['target'].count()

#'target' sütunundaki değerlerin ortalamasından daha büyük olan hastaların yaşları
df.loc[df['target'] > df['target'].mean(), 'age'].head()

# bağımsız değişkenleri grupladım

sensor = df.iloc[:,0:13]
sensor.head()

sensor.columns  #bağımsız değikenlerdeki kolonlar

from matplotlib import pyplot as plt
import seaborn as sns


def num_summary(sensor, num_var, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]  # aykırı değer yok
    print(sensor[num_var].describe(quantiles).T)  # numeric kolonlarlara göre yazdır

    if plot:
        sensor[num_var].hist()  # numeric kolonların title'ını alıcak
        plt.xlabel(num_var)
        plt.title(num_var)
        plt.show(block=True)


# bağımsız değişkenlerin  analizini yapmak için grafik oluşturduk
# tek tek çalıştırmak için bir fonk oluşturarak bu değişkenlerin grafiklerini çıkartıyorum
# sensor[num_var].describe(quantiles).T: Belirtilen bağımsız değişkenin özet istatistiklerini ekrana yazdırır.

num_summary(sensor , "age" ,plot = True) #age için

# tüm değişkenler için bir kod ile grafik üretme
for col in sensor:
        num_summary(sensor , col , plot=True)

##resting blood pressure ö gruplandırarak ortalama üzerindeki dağılımı gösteriyor
df.groupby('target')['restecg'].mean()

df.groupby('target')[['thalach']].mean().sort_values(by = "thalach")


def target_summ_with_num(dataframe, income, num_col):
    print(dataframe.groupby(['target']).agg({num_col: 'mean'}), end='\n\n\n')

#DataFrame'deki kategorik bir değişkenin farklı gruplarına göre bir veya daha fazla sayısal değişkenin özet istatistikleri

for col in sensor:
    target_summ_with_num(df,'target' , col)

# koreasyon hesaplama
corr = sensor.corr()
sns.set(rc = {'figure.figsize' : (19,10)})
sns.heatmap(corr , cmap='YlGnBu' ,annot=True)
plt.show()

#bağımlı ve bağımsız degiskenleri tanımladım
X = df.drop(columns='target', axis=1)
Y = df['target']
print(X)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Lojistik Regresyon kullanılarak bu veri kümeleri üzerinde bir model eğitilir ve doğruluk skoru hesaplanır.
# veri setini eğitim ve test veri kümelerine ayırdım
X_train , X_test , Y_train , Y_test = train_test_split(X,Y , test_size=0.2 , stratify =Y , random_state=2)

print(X.shape , X_train.shape , X_test.shape) #splitting the data into  training data & test data


#Model Training
#Logistic Regression

#Logistic regresyonu eğit
model = LogisticRegression()
model.fit(X_train , Y_train )

#Model Evaluation
#Eğitim datasının doğruluğu
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , Y_train)
print('accuracy on training data :  ' , training_data_accuracy)

##Accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction ,Y_test )

print('Accuracy on Test data :', test_data_accuracy)
#accuracy_score fonksiyonu,modelin yaptığı tahminlerle gerçek değerler arasındaki doğruluk oranını hesaplar.

##Building a predictive system
#eğitilmiş bir Lojistik Regresyon modelini kullanarak yeni bir veri noktası için tahmin yaptım
import numpy as np
input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)



input_data_as_numpy_array = np.asarray(input_data) #input dataı numpy'e çevirdim
input_data_reshaped= input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)