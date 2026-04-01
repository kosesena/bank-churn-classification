# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

> **Karar Ağacı (Decision Tree)** ve **K-En Yakın Komşu (KNN)** algoritmaları kullanarak banka müşterilerinin churn (ayrılma) durumunu tahmin eden makine öğrenmesi projesi.

---

## 📋 İçindekiler

- [Proje Hakkında](#-proje-hakkında)
- [Veri Seti](#-veri-seti)
- [Kullanılan Teknikler](#-kullanılan-teknikler)
- [Proje Akışı](#-proje-akışı)
- [Model Sonuçları](#-model-sonuçları)
- [Kurulum ve Çalıştırma](#-kurulum-ve-çalıştırma)
- [Proje Yapısı](#-proje-yapısı)

---

## 🎯 Proje Hakkında

Bankalar için müşteri kaybı (churn), gelir ve büyüme açısından kritik bir sorundur. Bu projede, bir bankanın **10.000 müşteri** verisini kullanarak hangi müşterilerin bankayı terk edeceğini önceden tahmin eden sınıflandırma modelleri geliştirilmiştir.

### ❓ Problem Tanımı

| | |
|---|---|
| **Hedef** | Müşterinin bankadan ayrılıp ayrılmayacağını tahmin etmek |
| **Hedef Değişken** | `Exited` → `1` = Ayrıldı, `0` = Kaldı |
| **Yaklaşım** | Gözetimli Öğrenme (Supervised Learning) - Sınıflandırma |
| **Algoritmalar** | Decision Tree, K-Nearest Neighbors (KNN) |

---

## 📊 Veri Seti

**Kaynak:** Bank Customer Churn Dataset
**Boyut:** 10.000 satır × 14 sütun

### Öznitelikler (Features)

| Öznitelik | Açıklama | Tip |
|-----------|----------|-----|
| `CreditScore` | Müşterinin kredi puanı | Sayısal |
| `Geography` | Ülke (France, Spain, Germany) | Kategorik |
| `Gender` | Cinsiyet (Male, Female) | Kategorik |
| `Age` | Yaş | Sayısal |
| `Tenure` | Bankadaki yıl sayısı | Sayısal |
| `Balance` | Hesap bakiyesi | Sayısal |
| `NumOfProducts` | Kullanılan ürün sayısı | Sayısal |
| `HasCrCard` | Kredi kartı var mı? (1/0) | Binary |
| `IsActiveMember` | Aktif üye mi? (1/0) | Binary |
| `EstimatedSalary` | Tahmini maaş | Sayısal |
| **`Exited`** | **Ayrıldı mı? (1/0) — Hedef Değişken** | **Binary** |

### Sınıf Dağılımı

```
┌─────────────────────────────────────────┐
│  Kaldı (0)   ████████████████████  ~80% │
│  Ayrıldı (1) █████                ~20% │
└─────────────────────────────────────────┘
```

> ⚠️ Veri seti **dengesizdir** (~%20 churn). Bu durum `class_weight='balanced'` parametresi ile ele alınmıştır.

---

## 🛠️ Kullanılan Teknikler

### Veri Ön İşleme

| Teknik | Neden? |
|--------|--------|
| 🔤 **One-Hot Encoding** | `Geography` ve `Gender` nominal değişkenler — sıralama anlamı taşımadığı için Ordinal Encoding yerine tercih edildi |
| 📏 **StandardScaler** | KNN mesafe tabanlı bir algoritma — farklı ölçeklerdeki değişkenler (ör. `Balance` vs `Age`) normalize edilmeli |
| ✂️ **Stratified Split** | Train/test ayırımında sınıf oranı korunarak (%67 eğitim, %33 test) bölündü |

### Model Geliştirme

| Teknik | Açıklama |
|--------|----------|
| 🌳 **Decision Tree + GridSearchCV** | `max_depth`, `criterion`, `min_samples_split`, `min_samples_leaf` parametreleri optimize edildi |
| 🏘️ **KNN + Elbow Yöntemi** | K=1'den K=20'ye kadar denenerek optimal komşu sayısı belirlendi |
| ⚖️ **class_weight='balanced'** | Azınlık sınıfına (churn) daha fazla ağırlık vererek dengesizlik problemi çözüldü |
| 🔄 **5-Fold Cross Validation** | Modelin farklı veri bölümlerinde tutarlılığı test edildi |

---

## 📈 Proje Akışı

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Veri Yükleme │────▶│     EDA      │────▶│  Veri Ön İşleme  │
│  (CSV)       │     │ (Görsellerle)│     │  (Encoding,      │
│              │     │              │     │   Scaling, Split) │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                   │
                          ┌────────────────────────┤
                          ▼                        ▼
                 ┌─────────────────┐     ┌─────────────────┐
                 │  Karar Ağacı    │     │      KNN        │
                 │  + GridSearchCV │     │  + Elbow Method │
                 └────────┬────────┘     └────────┬────────┘
                          │                        │
                          ▼                        ▼
                 ┌─────────────────────────────────────────┐
                 │         Model Karşılaştırması           │
                 │  (Accuracy, F1, Confusion Matrix, CV)   │
                 └─────────────────────────────────────────┘
```

### Adım Adım

1. **Veri Yükleme** — `bank_churn.csv` dosyası okunur
2. **Keşifsel Veri Analizi (EDA)** — Churn dağılımı, ülke ve cinsiyete göre analizler
3. **Veri Ön İşleme** — Gereksiz sütunların çıkarılması, One-Hot Encoding, ölçeklendirme
4. **Karar Ağacı Modeli** — GridSearchCV ile en iyi hiperparametreler bulunur
5. **KNN Modeli** — Elbow yöntemi ile optimal K belirlenir, ölçeklendirilmiş veri kullanılır
6. **Değerlendirme** — Confusion Matrix, Classification Report, Cross-Validation
7. **Karşılaştırma** — İki modelin performansı tablo ve görsel olarak karşılaştırılır

---

## 🏆 Model Sonuçları

### Karar Ağacı (Decision Tree)

- ✅ Hiperparametreler **GridSearchCV** ile optimize edildi
- ✅ **class_weight='balanced'** ile dengesiz sınıf yönetimi
- ✅ 5-Fold Cross Validation ile doğrulandı
- 📊 Öznitelik önemlilikleri (feature importance) görselleştirildi
- 🌳 Ağaç yapısı matplotlib ile çizildi

### KNN (K-En Yakın Komşu)

- ✅ **StandardScaler** ile ölçeklendirme yapıldı
- ✅ **Elbow yöntemi** ile optimal K değeri belirlendi
- ✅ 5-Fold Cross Validation ile doğrulandı

### Değerlendirme Metrikleri

Her iki model için aşağıdaki metrikler hesaplandı:

| Metrik | Açıklama |
|--------|----------|
| **Accuracy** | Genel doğruluk oranı |
| **Precision** | Pozitif tahminlerin ne kadarı doğru |
| **Recall** | Gerçek pozitiflerin ne kadarı yakalandı |
| **F1-Score** | Precision ve Recall'un harmonik ortalaması |
| **CV Score** | 5-Fold Cross Validation ortalaması |

---

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Çalıştırma

```bash
# Repo'yu klonla
git clone https://github.com/kosesena/bank-churn-classification.git
cd bank-churn-classification

# Jupyter Notebook'u başlat
jupyter notebook KARAR_AGACI_BANK_CHURN.ipynb
```

> 💡 **Google Colab** üzerinde de çalıştırabilirsiniz — notebook'u Colab'a yükleyip `bank_churn.csv` dosyasını upload edin.

---

## 📁 Proje Yapısı

```
bank-churn-classification/
│
├── KARAR_AGACI_BANK_CHURN.ipynb   # Ana notebook (tüm analiz ve modeller)
├── bank_churn.csv                  # Veri seti (10.000 müşteri kaydı)
└── README.md                       # Proje dokümantasyonu
```

---

## 🔑 Önemli Çıkarımlar

- 📌 **Dengesiz veri setlerinde** sadece Accuracy'ye bakmak yanıltıcıdır — F1-Score ve Recall daha güvenilir metriklerdir
- 📌 **KNN için ölçeklendirme şarttır** — farklı ölçeklerdeki değişkenler mesafe hesabını bozar
- 📌 **One-Hot Encoding**, nominal kategorik değişkenler için Ordinal Encoding'den daha uygundur
- 📌 **Cross-Validation** tek bir train/test split'e güvenmekten daha güvenilir sonuçlar verir
- 📌 **Hiperparametre optimizasyonu** (GridSearchCV) model performansını önemli ölçüde artırabilir

---

<p align="center">
  <b>⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!</b>
</p>
