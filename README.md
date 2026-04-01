<div align="center">

# 🏦 Bank Customer Churn Prediction

### Banka Müşteri Kayıp Tahmini — Makine Öğrenmesi ile Sınıflandırma

<br>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)

<br>

<table>
<tr>
<td align="center"><b>10.000</b><br><sub>Müşteri Verisi</sub></td>
<td align="center"><b>2</b><br><sub>ML Algoritması</sub></td>
<td align="center"><b>840</b><br><sub>Model Eğitildi</sub></td>
<td align="center"><b>5-Fold</b><br><sub>Cross Validation</sub></td>
</tr>
</table>

</div>

<br>

---

<br>

## 🤔 Bu Projede Ne Yapıyoruz?

Bir bankanın **10.000 müşteri verisini** makineye veriyoruz ve diyoruz ki:

> *"Bu verilere bak, örüntüleri öğren. Sonra sana yeni bir müşteri verdiğimde bana söyle:*
> ***bu müşteri bankadan ayrılacak mı, kalmaya devam mı edecek?"***

İşte buna **Makine Öğrenmesi** diyoruz. Makine geçmiş verilerden kalıplar öğreniyor ve gelecekteki müşteriler hakkında tahmin yapıyor.

<br>

### 💡 Gerçek Hayatta Ne İşe Yarar?

Banka, ayrılacak müşteriyi **önceden** bilirse:

<table>
<tr>
<td>🎯</td><td>Özel kampanya sunabilir</td>
<td>💰</td><td>Faiz indirimi yapabilir</td>
</tr>
<tr>
<td>👤</td><td>Kişisel danışman atayabilir</td>
<td>🛡️</td><td>Müşteriyi kaybetmeden önlem alabilir</td>
</tr>
</table>

> **Yeni müşteri kazanmak, mevcut müşteriyi tutmaktan 5-7 kat daha pahalıdır.** Bu yüzden churn tahmini bankalar için kritiktir.

<br>

---

<br>

## 📊 Veri Seti

<table>
<tr>
<td><b>Kaynak</b></td>
<td><a href="https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn">Bank Customer Churn Dataset — Kaggle</a></td>
</tr>
<tr>
<td><b>Boyut</b></td>
<td>10.000 satır × 14 sütun</td>
</tr>
</table>

<br>

### Öznitelikler

| Öznitelik | Açıklama | Tip | Min | Ortalama | Max |
|:---|:---|:---:|---:|---:|---:|
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

<br>

### Kategorik Değişken Dağılımları

<table>
<tr><th colspan="2">🌍 Ülke Dağılımı</th><th colspan="2">👥 Cinsiyet Dağılımı</th></tr>
<tr>
<td>🇫🇷 France</td><td><code>4.928</code> (%49.3)</td>
<td>👨 Male</td><td><code>5.494</code> (%54.9)</td>
</tr>
<tr>
<td>🇩🇪 Germany</td><td><code>2.565</code> (%25.7)</td>
<td>👩 Female</td><td><code>4.506</code> (%45.1)</td>
</tr>
<tr>
<td>🇪🇸 Spain</td><td><code>2.507</code> (%25.1)</td>
<td></td><td></td>
</tr>
</table>

<br>

### Hedef Değişken Dağılımı

```
Kaldı (0):    7.256 müşteri  (%72.6)  ████████████████████████████░░░░░░░░
Ayrıldı (1):  2.744 müşteri  (%27.4)  ██████████░░░░░░░░░░░░░░░░░░░░░░░░
```

> ⚠️ Veri **dengesiz**: ayrılan müşteri sayısı çok daha az. Modelin "herkes kalır" deyip %73 accuracy almasını engellemek için `class_weight='balanced'` kullandık.

<br>

### Churn Oranları — Kırılımlar

<table>
<tr><th>Kırılım</th><th>Churn Oranı</th><th>Durum</th></tr>
<tr><td>🇫🇷 France</td><td align="center">%24.1</td><td>🟢 Düşük risk</td></tr>
<tr><td>🇩🇪 Germany</td><td align="center"><b>%35.9</b></td><td>🔴 Yüksek risk</td></tr>
<tr><td>🇪🇸 Spain</td><td align="center">%25.4</td><td>🟢 Düşük risk</td></tr>
<tr><td>👩 Female</td><td align="center"><b>%29.3</b></td><td>🟡 Orta risk</td></tr>
<tr><td>👨 Male</td><td align="center">%25.9</td><td>🟢 Düşük risk</td></tr>
</table>

> 📌 **Almanya'daki müşteriler** ve **kadın müşteriler** daha yüksek churn oranına sahip.

<br>

---

<br>

## 🧠 Kullanılan Algoritmalar

<br>

<table>
<tr>
<td width="50%">

### 🌳 Karar Ağacı (Decision Tree)

Verideki kalıpları **"eğer-ise" kuralları** şeklinde öğrenir:

```
Yaş > 42 mi?
├── Evet → Ürün sayısı > 2 mi?
│   ├── Evet → 🔴 AYRILACAK
│   └── Hayır → Almanya'da mı?
│       ├── Evet → 🔴 AYRILACAK
│       └── Hayır → 🟢 KALACAK
└── Hayır → 🟢 KALACAK
```

</td>
<td width="50%">

### 🏘️ KNN (K-En Yakın Komşu)

Yeni müşteriyi tahmin ederken **en benzer K müşteriyi** bulup çoğunluğa göre karar verir:

```
Yeni müşteri → En yakın 14 komşu
├── 10 tanesi "kaldı"
├── 4 tanesi "ayrıldı"
└── Sonuç: 🟢 KALACAK (çoğunluk)
```

</td>
</tr>
</table>

<br>

---

<br>

## ⚙️ Hiperparametre Seçimleri

<br>

### 🌳 Karar Ağacı — GridSearchCV ile Optimize Edildi

> 168 farklı parametre kombinasyonu × 5 fold = **840 model eğitildi**, en iyisi seçildi.

| Parametre | Değer | Açıklama |
|:---|:---:|:---|
| `max_depth` | **3** | Ağaç en fazla 3 seviye derine iner — fazla derin = ezberleme (overfitting) |
| `criterion` | **gini** | Dallanma kararı Gini impurity ile yapılır |
| `min_samples_split` | **2** | Bir düğümün bölünmesi için en az 2 örnek gerekli |
| `min_samples_leaf` | **10** | Her yaprak düğümde en az 10 örnek olmalı — aşırı öğrenmeyi engeller |
| `class_weight` | **balanced** | Azınlık sınıfına (ayrılan) daha fazla ağırlık verir |

<details>
<summary><b>🔍 GridSearchCV nedir?</b> (tıkla)</summary>
<br>
Tüm parametre kombinasyonlarını dener, her birini 5 kez farklı veri dilimleriyle test eder ve en iyi sonucu verenini otomatik seçer. Bu projede <code>max_depth</code> (7 değer) × <code>criterion</code> (2 değer) × <code>min_samples_split</code> (4 değer) × <code>min_samples_leaf</code> (3 değer) = 168 kombinasyon × 5 fold = <b>840 model</b> eğitildi.
</details>

<br>

### 🏘️ KNN — Elbow Yöntemi ile K Belirlendi

> K=1'den K=20'ye kadar her değer 5-Fold CV ile test edildi.

| Parametre | Değer | Açıklama |
|:---|:---:|:---|
| `n_neighbors (K)` | **14** | Tahmin yaparken en yakın 14 komşuya bakılır |
| `Scaling` | **StandardScaler** | Tüm değişkenler aynı ölçeğe getirildi — KNN için zorunlu |

<details>
<summary><b>🔍 Neden ölçeklendirme (scaling) şart?</b> (tıkla)</summary>
<br>
KNN mesafe tabanlı çalışır. <code>Balance = 100.000€</code> ve <code>Age = 42</code> olduğunda, ölçeklendirme yapılmazsa KNN sadece Balance'a bakar, Age'i görmezden gelir. <b>StandardScaler</b> her değişkeni ortalama=0, standart sapma=1 olacak şekilde dönüştürür, böylece tüm öznitelikler eşit ağırlıkta değerlendirilir.
</details>

<br>

---

<br>

## 📈 Test ve Eğitim Yapısı

```
10.000 Müşteri
     │
     ├── %67 → Eğitim Seti (6.700 müşteri) → Model bununla öğreniyor
     │
     └── %33 → Test Seti (3.300 müşteri) → Model bunu HİÇ görmedi, sınav gibi
```

Ek olarak **5-Fold Cross Validation** uygulandı — tek bir split'e güvenmek yerine 5 farklı bölüm denendi:

```
Eğitim verisi 5 parçaya bölünür:

  Tur 1:  [🔵 TEST]  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]   → Skor 1
  Tur 2:  [⬜ Eğit]  [🔵 TEST]  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]   → Skor 2
  Tur 3:  [⬜ Eğit]  [⬜ Eğit]  [🔵 TEST]  [⬜ Eğit]  [⬜ Eğit]   → Skor 3
  Tur 4:  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]  [🔵 TEST]  [⬜ Eğit]   → Skor 4
  Tur 5:  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]  [⬜ Eğit]  [🔵 TEST]   → Skor 5

  Sonuç = 5 skorun ortalaması → daha güvenilir değerlendirme
```

<br>

---

<br>

## 🏆 Sonuçlar

<br>

<div align="center">

### Karar Ağacı 🌳

</div>

| Metrik | Değer | Açıklama |
|:---|:---:|:---|
| **Test Accuracy** | `%61.3` | 3.300 müşteriden 2.024'ünü doğru tahmin etti |
| **CV Accuracy** | `%60.1 ± 1.3` | 5-Fold ortalaması — güvenilir skor |
| **F1 Score (Ayrılan)** | `0.48` | Precision ve Recall'un dengesi |
| **Recall (Ayrılan)** | `%64` | Gerçekten ayrılan 906 kişiden **584'ünü yakaladı** |
| **Precision (Ayrılan)** | `%38` | "Ayrılacak" dediklerinin %38'i gerçekten ayrıldı |

**Confusion Matrix:**
```
                      Tahmin Edilen
                   Kaldı       Ayrıldı
                ┌───────────┬───────────┐
  Gerçek Kaldı  │   1.440   │    954    │  ← 954 yanlış alarm
                ├───────────┼───────────┤
  Gerçek Ayrıldı│    322    │    584    │  ← 584 doğru yakalandı ✓
                └───────────┴───────────┘
```

> 📌 `class_weight='balanced'` sayesinde ayrılan müşterilerin **%64'ü yakalandı**, ancak yanlış alarm oranı yüksek (954 kişi).

<br>

<div align="center">

### KNN 🏘️ (K=14)

</div>

| Metrik | Değer | Açıklama |
|:---|:---:|:---|
| **Test Accuracy** | `%71.9` | 3.300 müşteriden 2.373'ünü doğru tahmin etti |
| **CV Accuracy** | `%72.5 ± 0.4` | 5-Fold ortalaması — güvenilir skor |
| **F1 Score (Ayrılan)** | `0.08` | Çok düşük — ayrılanları neredeyse hiç bulamıyor |
| **Recall (Ayrılan)** | `%5` | Gerçekten ayrılan 906 kişiden **sadece 43'ünü yakaladı** |
| **Precision (Ayrılan)** | `%40` | "Ayrılacak" dediklerinin %40'ı gerçekten ayrıldı |

**Confusion Matrix:**
```
                      Tahmin Edilen
                   Kaldı       Ayrıldı
                ┌───────────┬───────────┐
  Gerçek Kaldı  │   2.330   │     64    │  ← Kalanları çok iyi biliyor
                ├───────────┼───────────┤
  Gerçek Ayrıldı│    863    │     43    │  ← Ayrılanları neredeyse hiç bulamadı ✗
                └───────────┴───────────┘
```

> 📌 KNN accuracy olarak yüksek (%72) ama **ayrılacak müşterileri neredeyse hiç yakalayamıyor**. Pratikte "herkes kalır" diyen bir modelden farklı değil.

<br>

---

<br>

## 🔍 Hangi Model Kazandı?

<div align="center">

| Metrik | Karar Ağacı 🌳 | KNN 🏘️ | Kazanan |
|:---|:---:|:---:|:---:|
| **Accuracy** | %61.3 | %71.9 | KNN |
| **CV Accuracy** | %60.1 | %72.5 | KNN |
| **Recall (Ayrılan)** | **%64** | %5 | **Karar Ağacı** |
| **F1 (Ayrılan)** | **0.48** | 0.08 | **Karar Ağacı** |
| **Yakalanan Churn** | **584 / 906** | 43 / 906 | **Karar Ağacı** |

</div>

<br>

<table>
<tr>
<td width="50%" align="center">
<h3>📊 Accuracy'ye bakarsak</h3>
<b>KNN kazanır</b> (%72 vs %61)
<br><br>
<sub>Ama bu yanıltıcı — çünkü KNN neredeyse herkese "kalır" diyor</sub>
</td>
<td width="50%" align="center">
<h3>🎯 Asıl amaca bakarsak</h3>
<b>Karar Ağacı kazanır!</b>
<br><br>
<sub>Ayrılacak müşterilerin %64'ünü yakalıyor,<br>KNN sadece %5'ini yakalayabiliyor</sub>
</td>
</tr>
</table>

> ⚠️ **Önemli ders:** Dengesiz veri setlerinde **Accuracy yanıltıcıdır!** Asıl bakılması gereken metrikler **F1-Score** ve **Recall**'dur.

<br>

---

<br>

## 🔑 En Etkili Öznitelikler

Karar ağacı hangi bilgilere bakarak karar veriyor?

```
Age (Yaş)              ██████████████████████████████████████████  %39.6
NumOfProducts (Ürün)   ████████████████████████████               %27.7
Geography_Germany      ███████████████████████████                %27.3
IsActiveMember         ████                                       %4.4
CreditScore            █                                          %1.0
```

<table>
<tr>
<td>🔴 <b>Yaş</b> — en belirleyici faktör, yaşlı müşteriler daha çok ayrılıyor</td>
</tr>
<tr>
<td>🔴 <b>Ürün sayısı</b> — çok fazla veya çok az ürün kullananlar risk altında</td>
</tr>
<tr>
<td>🔴 <b>Almanya'da olmak</b> — Alman müşterilerin churn oranı %35.9 ile en yüksek</td>
</tr>
<tr>
<td>🟢 <b>Balance, Salary</b> — finansal değişkenler karar ağacında etkisiz çıktı</td>
</tr>
</table>

<br>

---

<br>

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

> 💡 Google Colab üzerinde de çalıştırabilirsiniz — notebook'u Colab'a yükleyip `bank_churn.csv` dosyasını upload edin.

<br>

---

<br>

## 📁 Proje Yapısı

```
bank-churn-classification/
│
├── 📓 KARAR_AGACI_BANK_CHURN.ipynb   → Ana notebook (analiz + modeller)
├── 📄 bank_churn.csv                  → Veri seti (10.000 müşteri)
└── 📋 README.md                       → Proje dokümantasyonu
```
