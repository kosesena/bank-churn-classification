# 🏦 Bank Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

---

## 🤔 Bu Projede Ne Yapıyoruz?

Bir bankanın **10.000 müşteri verisini** makineye veriyoruz ve diyoruz ki:

> "Bu verilere bak, öğren. Sonra sana yeni bir müşteri verdiğimde bana söyle: **bu müşteri bankadan ayrılacak mı, kalmaya devam mı edecek?**"

İşte buna **Makine Öğrenmesi** diyoruz. Makine geçmiş verilerden örüntüler öğreniyor ve gelecekteki müşteriler hakkında tahmin yapıyor.

### 💡 Gerçek Hayatta Ne İşe Yarar?

Banka, ayrılacak müşteriyi **önceden** bilirse:
- Özel kampanya sunabilir
- Faiz indirimi yapabilir
- Kişisel danışman atayabilir
- Müşteriyi kaybetmeden önlem alabilir

**Yeni müşteri kazanmak, mevcut müşteriyi tutmaktan 5-7 kat daha pahalıdır.** Bu yüzden churn tahmini bankalar için kritiktir.

---

## 📊 Veri Seti

**Kaynak:** [Bank Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
**Boyut:** 10.000 satır × 14 sütun

### Öznitelikler

| Öznitelik | Açıklama | Tip | Min | Ortalama | Max |
|-----------|----------|:---:|:---:|:--------:|:---:|
| `CreditScore` | Kredi puanı | Sayısal | 350 | 648 | 850 |
| `Geography` | Ülke | Kategorik | — | — | — |
| `Gender` | Cinsiyet | Kategorik | — | — | — |
| `Age` | Yaş | Sayısal | 18 | 39 | 74 |
| `Tenure` | Bankadaki yıl sayısı | Sayısal | 0 | 5 | 10 |
| `Balance` | Hesap bakiyesi (€) | Sayısal | 0 | 50.338 | 260.000 |
| `NumOfProducts` | Kullanılan ürün sayısı | Sayısal | 1 | 1.5 | 4 |
| `HasCrCard` | Kredi kartı var mı? | Binary | 0 | 0.71 | 1 |
| `IsActiveMember` | Aktif üye mi? | Binary | 0 | 0.51 | 1 |
| `EstimatedSalary` | Tahmini maaş (€) | Sayısal | 60 | 99.373 | 199.983 |
| **`Exited`** | **Ayrıldı mı? (Hedef)** | **Binary** | **0** | **0.27** | **1** |

### Kategorik Değişken Dağılımları

**Ülke:**
```
France:   4.928 müşteri  (%49.3)  █████████████████████████
Germany:  2.565 müşteri  (%25.7)  █████████████
Spain:    2.507 müşteri  (%25.1)  ████████████
```

**Cinsiyet:**
```
Male:     5.494 müşteri  (%54.9)  ████████████████████████████
Female:   4.506 müşteri  (%45.1)  ██████████████████████
```

### Hedef Değişken (Exited) Dağılımı

```
Kaldı (0):    7.256 müşteri  (%72.6)  ████████████████████████████░░░░
Ayrıldı (1):  2.744 müşteri  (%27.4)  ██████████░░░░░░░░░░░░░░░░░░░░░
```

> ⚠️ Veri **dengesiz**: ayrılan müşteri sayısı çok daha az. Bu, modelin "herkes kalır" deyip %73 accuracy almasına yol açabilir. Bu sorunu `class_weight='balanced'` ile çözdük.

### Churn Oranları (Kırılımlar)

| Kırılım | Churn Oranı |
|---------|:-----------:|
| 🇫🇷 France | %24.1 |
| 🇩🇪 Germany | **%35.9** |
| 🇪🇸 Spain | %25.4 |
| 👩 Female | **%29.3** |
| 👨 Male | %25.9 |

> 📌 Almanya'daki müşteriler ve kadın müşteriler daha yüksek churn oranına sahip.

---

## 🧠 Hangi Algoritmaları Kullandık?

### 1. 🌳 Karar Ağacı (Decision Tree)

Karar ağacı, verideki kalıpları **"eğer-ise" kuralları** şeklinde öğrenir:

```
Müşterinin yaşı > 42 mi?
├── Evet → Ürün sayısı > 2 mi?
│   ├── Evet → 🔴 AYRILACAK
│   └── Hayır → Almanya'da mı?
│       ├── Evet → 🔴 AYRILACAK
│       └── Hayır → 🟢 KALACAK
└── Hayır → 🟢 KALACAK
```

### 2. 🏘️ KNN (K-En Yakın Komşu)

KNN, yeni bir müşteriyi tahmin ederken **en benzer K müşteriyi** bulur ve çoğunluğa göre karar verir:

```
Yeni müşteri geldi → En yakın 14 müşteriye bak
├── 10 tanesi "kaldı"
├── 4 tanesi "ayrıldı"
└── Sonuç: 🟢 KALACAK (çoğunluk)
```

---

## ⚙️ Hiperparametre Seçimleri (Model Ayarları)

### Karar Ağacı — GridSearchCV ile Optimize Edildi

168 farklı parametre kombinasyonu denendi, en iyisi:

| Parametre | Seçilen Değer | Ne Anlama Geliyor? |
|-----------|:---:|---|
| `max_depth` | **3** | Ağaç en fazla 3 seviye derine iner. Fazla derin = ezberleme (overfitting) |
| `criterion` | **gini** | Dallanma kararı Gini impurity ile yapılır (alternatif: entropy) |
| `min_samples_split` | **2** | Bir düğümün bölünmesi için en az 2 örnek gerekli |
| `min_samples_leaf` | **10** | Her yaprak düğümde en az 10 örnek olmalı (aşırı öğrenmeyi engeller) |
| `class_weight` | **balanced** | Azınlık sınıfına (ayrılan) daha fazla ağırlık verir |

> 🔍 **GridSearchCV nedir?** Tüm parametre kombinasyonlarını dener, her birini 5 kez farklı veri dilimleriyle test eder ve en iyi sonucu verenini seçer. Toplam 168 × 5 = **840 model** eğitildi.

### KNN — Elbow Yöntemi ile K Belirlendi

K=1'den K=20'ye kadar her değer denendi:

| Parametre | Seçilen Değer | Ne Anlama Geliyor? |
|-----------|:---:|---|
| `n_neighbors (K)` | **14** | Tahmin yaparken en yakın 14 komşuya bakılır |
| `Scaling` | **StandardScaler** | Tüm değişkenler aynı ölçeğe getirildi (KNN için zorunlu) |

> 🔍 **Neden ölçeklendirme?** Balance 100.000€, Age 42 olabilir. Ölçeklendirme yapılmazsa KNN sadece Balance'a bakar, Age'i görmezden gelir. StandardScaler her değişkeni ortalama=0, standart sapma=1 olacak şekilde dönüştürür.

---

## 📈 Test ve Eğitim Nasıl Yapıldı?

```
10.000 Müşteri
     │
     ├── %67 → Eğitim Seti (6.700 müşteri) → Model bununla öğreniyor
     │
     └── %33 → Test Seti (3.300 müşteri) → Model bunu HİÇ görmedi, sınav gibi
```

Ayrıca **5-Fold Cross Validation** uygulandı:

```
Eğitim verisi 5 parçaya bölünür:
  Tur 1: [TEST] [Eğit] [Eğit] [Eğit] [Eğit]  → Skor 1
  Tur 2: [Eğit] [TEST] [Eğit] [Eğit] [Eğit]  → Skor 2
  Tur 3: [Eğit] [Eğit] [TEST] [Eğit] [Eğit]  → Skor 3
  Tur 4: [Eğit] [Eğit] [Eğit] [TEST] [Eğit]  → Skor 4
  Tur 5: [Eğit] [Eğit] [Eğit] [Eğit] [TEST]  → Skor 5

  Sonuç = 5 skorun ortalaması (daha güvenilir)
```

---

## 🏆 Sonuçlar

### Karar Ağacı Sonuçları

| Metrik | Değer | Açıklama |
|--------|:-----:|----------|
| **Accuracy** | **%61.3** | 3300 müşteriden 2024'ünü doğru tahmin etti |
| **CV Accuracy** | **%60.1** | 5-Fold ortalaması (güvenilir skor) |
| **F1 (Ayrılan)** | **0.48** | Ayrılan müşterileri yakalama başarısı |
| **Recall (Ayrılan)** | **%64** | Gerçekten ayrılan 906 kişiden 584'ünü yakaladı |
| **Precision (Ayrılan)** | **%38** | "Ayrılacak" dediklerinin %38'i gerçekten ayrıldı |

**Confusion Matrix (Karışıklık Matrisi):**

```
                    Tahmin
                Kaldı    Ayrıldı
Gerçek  Kaldı  [ 1440      954 ]    ← 954 kişiyi yanlışlıkla "ayrılacak" dedi
       Ayrıldı [  322      584 ]    ← 584 ayrılanı doğru yakaladı ✓
```

> 📌 Karar ağacı `class_weight='balanced'` sayesinde ayrılan müşterilerin **%64'ünü** yakaladı. Ama bunun karşılığında çok fazla yanlış alarm veriyor (954 kişi).

### KNN Sonuçları (K=14)

| Metrik | Değer | Açıklama |
|--------|:-----:|----------|
| **Accuracy** | **%71.9** | 3300 müşteriden 2373'ünü doğru tahmin etti |
| **CV Accuracy** | **%72.5** | 5-Fold ortalaması (güvenilir skor) |
| **F1 (Ayrılan)** | **0.08** | Ayrılan müşterileri yakalama başarısı çok düşük |
| **Recall (Ayrılan)** | **%5** | Gerçekten ayrılan 906 kişiden sadece 43'ünü yakaladı |
| **Precision (Ayrılan)** | **%40** | "Ayrılacak" dediklerinin %40'ı gerçekten ayrıldı |

**Confusion Matrix:**

```
                    Tahmin
                Kaldı    Ayrıldı
Gerçek  Kaldı  [ 2330       64 ]    ← Kalanları çok iyi biliyor
       Ayrıldı [  863       43 ]    ← Ayrılanların neredeyse hepsini kaçırdı ✗
```

> 📌 KNN accuracy olarak daha yüksek (%72 vs %61) ama **ayrılacak müşterileri neredeyse hiç yakalayamıyor**. "Herkes kalır" diyen bir modelden çok da farklı değil.

---

## 🔍 Karşılaştırma: Hangi Model Daha İyi?

| | Karar Ağacı 🌳 | KNN 🏘️ |
|---|:---:|:---:|
| **Accuracy** | %61.3 | **%71.9** |
| **CV Accuracy** | %60.1 | **%72.5** |
| **Recall (Ayrılan)** | **%64** | %5 |
| **F1 (Ayrılan)** | **0.48** | 0.08 |
| **Ayrılanı yakalama** | **584 / 906** | 43 / 906 |

### Verdict

- **Accuracy'ye bakarsak:** KNN kazanır (%72 vs %61)
- **Asıl amaca bakarsak:** Karar Ağacı kazanır! Çünkü amacımız **ayrılacak müşteriyi bulmak**, ve karar ağacı bunların %64'ünü yakalarken, KNN sadece %5'ini yakalıyor.

> ⚠️ **Önemli ders:** Dengesiz veri setlerinde **Accuracy yanıltıcıdır!** F1-Score ve Recall asıl bakılması gereken metriklerdir.

---

## 🔑 En Etkili Öznitelikler (Karar Ağacı)

Karar ağacı hangi bilgilere bakarak karar veriyor?

```
Age (Yaş)               ████████████████████████████████████████  %39.6
NumOfProducts (Ürün)     ██████████████████████████████           %27.7
Geography_Germany        █████████████████████████████            %27.3
IsActiveMember           ████                                     %4.4
CreditScore              █                                        %1.0
```

**Yorumlar:**
- 🔴 **Yaş** en belirleyici faktör — yaşlı müşteriler daha çok ayrılıyor
- 🔴 **Ürün sayısı** — çok fazla veya çok az ürün kullananlar risk altında
- 🔴 **Almanya'da olmak** — Alman müşterilerin churn oranı daha yüksek
- 🟢 Balance, Salary gibi finansal değişkenler karar ağacında etkisiz çıktı

---

## 🚀 Kurulum ve Çalıştırma

```bash
# Gerekli kütüphaneler
pip install pandas numpy matplotlib seaborn scikit-learn

# Repo'yu klonla
git clone https://github.com/kosesena/bank-churn-classification.git
cd bank-churn-classification

# Jupyter Notebook'u başlat
jupyter notebook KARAR_AGACI_BANK_CHURN.ipynb
```

---

## 📁 Dosyalar

```
bank-churn-classification/
├── KARAR_AGACI_BANK_CHURN.ipynb   # Ana notebook (analiz + modeller)
├── bank_churn.csv                  # Veri seti (10.000 müşteri)
└── README.md                       # Bu dosya
```

