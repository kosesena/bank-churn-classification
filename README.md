<div align="center">

# 🏦 Bank Customer Churn Prediction

### Banka Müşteri Kayıp Tahmini — Karar Ağaçları ile Sınıflandırma

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

## 📚 Teorik Altyapı

<br>

### 🔷 Sınıflandırma Nedir?

Verinin içerdiği ortak özelliklere göre bir veri veya veri grubunun **hangi sınıfa dahil olduğunun belirlenmesi** işlemidir. Bir öğrenme algoritmasına dayanır ve amaç bir **sınıflandırma modelinin** yaratılmasıdır.

> *Örneğin: Müşteri bankadan "ayrılacak mı" yoksa "kalacak mı"?*

<br>

### 📋 Sınıflandırma Süreci

Sınıflandırma iki temel adımdan oluşur:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ADIM 1: MODEL OLUŞTURMA                                      │
│   ─────────────────────────                                     │
│   Eğitim verisi kullanılarak model eğitilir.                    │
│   Veri tabanındaki kayıtların nitelikleri analiz edilir.         │
│                                                                 │
│   Eğitim Verisi ──▶ [Sınıflandırma Algoritması] ──▶ MODEL      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ADIM 2: TEST VE TAHMİN                                       │
│   ───────────────────────                                       │
│   Model, hiç görmediği test verisi üzerinde sınanır.            │
│   Başarılıysa yeni veriler üzerinde tahmin yapabilir.           │
│                                                                 │
│   Test Verisi ──▶ [EĞİTİLMİŞ MODEL] ──▶ TAHMİN                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

<br>

---

<br>

### 🌳 Karar Ağaçları ile Sınıflandırma

Karar ağaçları, akış şemalarına benzeyen yapılardır. Her bir nitelik bir **düğüm** tarafından temsil edilir.

<table>
<tr>
<td align="center" width="33%">
<h4>🔵 Kök (Root)</h4>
<sub>En üstteki düğüm.<br>İlk dallanma buradan başlar.</sub>
</td>
<td align="center" width="33%">
<h4>🟡 Dal (Branch)</h4>
<sub>Kök ile yaprak arasında<br>kalan ara düğümler.</sub>
</td>
<td align="center" width="33%">
<h4>🟢 Yaprak (Leaf)</h4>
<sub>En son düğüm. Sınıf<br>kararının verildiği yer.</sub>
</td>
</tr>
</table>

```
                        [Yaş > 42]              ← 🔵 KÖK
                       /          \
                     Evet         Hayır
                     /               \
           [Ürün > 2]              🟢 KALACAK    ← YAPRAK
            /       \
         Evet      Hayır
          /           \
   🟢 AYRILACAK    [Almanya?]       ← 🟡 DAL
                    /       \
                  Evet     Hayır
                  /           \
          🟢 AYRILACAK    🟢 KALACAK  ← YAPRAKLAR
```

<br>

---

<br>

### 🔥 Dallanma Kriterleri

Karar ağaçlarında en önemli soru: **dallanma hangi niteliğe göre yapılacak?**

Her farklı kriter için bir algoritma geliştirilmiştir:

<table>
<tr>
<th>Algoritma Grubu</th>
<th>Yöntem</th>
<th>Bu Projede</th>
</tr>
<tr>
<td><b>Entropiye Dayalı</b></td>
<td>ID3, C4.5 → Bilgi Kazancı (Information Gain)</td>
<td><code>criterion='entropy'</code></td>
</tr>
<tr>
<td><b>Sınıflandırma ve Regresyon Ağaçları (CART)</b></td>
<td>Gini Impurity</td>
<td><code>criterion='gini'</code> ✅ seçildi</td>
</tr>
<tr>
<td><b>Bellek Tabanlı Sınıflandırma</b></td>
<td>K-En Yakın Komşu (KNN)</td>
<td><code>KNeighborsClassifier</code></td>
</tr>
</table>

<br>

---

<br>

### 📐 Entropi (Belirsizlik Ölçüsü)

Bir sistemdeki **belirsizliğin ölçüsüne** entropi denir. Entropi yüksekse belirsizlik fazla, düşükse sistem daha düzenlidir.

<div align="center">

**H(S) = − Σ pᵢ · log₂(pᵢ)**

</div>

<table>
<tr>
<td width="33%" align="center">
<h4>H = 0</h4>
<sub>Tüm örnekler aynı sınıfta<br>→ <b>Belirsizlik yok</b><br>Örn: [✓, ✓, ✓, ✓]</sub>
</td>
<td width="33%" align="center">
<h4>H = 1</h4>
<sub>Örnekler yarı yarıya<br>→ <b>Maksimum belirsizlik</b><br>Örn: [✓, ✓, ✗, ✗]</sub>
</td>
<td width="33%" align="center">
<h4>0 < H < 1</h4>
<sub>Bir sınıf baskın<br>→ <b>Kısmi belirsizlik</b><br>Örn: [✓, ✓, ✓, ✗]</sub>
</td>
</tr>
</table>

**Bu projede entropi hesabı:**

```
Veri setinde:  7.256 Kaldı, 2.744 Ayrıldı  (toplam 10.000)

p(Kaldı)   = 7256/10000 = 0.726
p(Ayrıldı) = 2744/10000 = 0.274

H(Exited) = −(0.726 × log₂(0.726)) − (0.274 × log₂(0.274))
H(Exited) = −(0.726 × (−0.462)) − (0.274 × (−1.868))
H(Exited) = 0.336 + 0.512
H(Exited) ≈ 0.848

→ Belirsizlik yüksek (1'e yakın), yani sınıflar çok net ayrışmıyor.
```

<br>

---

<br>

### 📊 Bilgi Kazancı (Information Gain)

Bir niteliğe göre dallanmanın **ne kadar bilgi kazandırdığını** ölçer. Hangi nitelik en çok belirsizliği azaltıyorsa, dallanma ondan yapılır.

<div align="center">

**Kazanç(X, T) = H(T) − H(X, T)**

</div>

> *H(T) = dallanmadan önceki entropi, H(X, T) = dallanmadan sonraki entropi*
>
> Fark ne kadar büyükse, o nitelik o kadar bilgilendirici demektir.

**Derste işlenen hava durumu örneği:**

```
Tablo 3.7 — Elde Edilen Kazanç Ölçütleri:

  Kazanç(HAVA)    = 0.246   ← 🏆 En yüksek → ilk dallanma buradan
  Kazanç(NEM)     = 0.152
  Kazanç(RÜZGAR)  = 0.048
  Kazanç(ISI)     = 0.029

  → HAVA niteliği en fazla bilgiyi sağlıyor, ağaç buradan başlar.
```

**Bu projede scikit-learn aynı mantığı kullanır:**

Ağaç her dallanma noktasında `Age`, `Balance`, `Geography` gibi tüm niteliklerin bilgi kazancını (veya Gini impurity'sini) hesaplar ve **en iyi ayrımı yapanı** seçer. Sonuç olarak **Age (%39.6)** en bilgilendirici nitelik çıkmıştır.

<br>

---

<br>

### 🔄 ID3 ve C4.5 Algoritmaları

<table>
<tr>
<td width="50%">

#### ID3 (Iterative Dichotomiser 3)

- Quinlan tarafından geliştirildi
- **Entropi** ve **bilgi kazancı** kullanır
- Her adımda en yüksek kazançlı niteliği seçer
- Sadece kategorik verilerle çalışır
- Eksik veri desteği yok

</td>
<td width="50%">

#### C4.5 (ID3'ün geliştirilmiş hali)

- **Kazanç oranı** (Gain Ratio) kullanır → çok dallı niteliklere karşı düzeltme
- Sürekli (sayısal) verilerle de çalışır
- **Eksik veri** desteği var
- Kazanç(X) = F × (H(T) − H(X,T))
- Budama (pruning) yapabilir

</td>
</tr>
</table>

<details>
<summary><b>🔍 Kazanç Oranı (Gain Ratio) nedir?</b> (tıkla)</summary>
<br>

Bilgi kazancı, çok fazla değere sahip nitelikleri (ör: müşteri ID — her kayıt farklı) haksız yere tercih edebilir. Kazanç oranı bunu düzeltir:

```
                    Kazanç(X)
Kazanç Oranı = ─────────────────
                 Ayrıştırma Bilgisi(X)
```

Ayrıştırma bilgisi, niteliğin kendi entropisidir. Çok fazla farklı değere sahip niteliklerde bu değer yüksek olur, dolayısıyla kazanç oranını düşürür.

</details>

<br>

---

<br>

### 📝 Karar Kuralları Oluşturma

Elde edilen karar ağacından **IF-THEN kuralları** çıkarılabilir. Bu kurallar programlama dillerindeki koşul yapılarına benzer.

**Bu projedeki karar ağacından çıkarılan kurallar:**

```
KURAL 1:  Eğer Yaş ≤ 42                                 → KALACAK
KURAL 2:  Eğer Yaş > 42  VE  Ürün Sayısı > 2            → AYRILACAK
KURAL 3:  Eğer Yaş > 42  VE  Ürün ≤ 2  VE  Almanya'da   → AYRILACAK
KURAL 4:  Eğer Yaş > 42  VE  Ürün ≤ 2  VE  Almanya değil → KALACAK
```

> Bu kurallar sayesinde model bir **kara kutu** olmaktan çıkar ve kararlarının **neden** verildiği anlaşılabilir. Bu, karar ağaçlarının en önemli avantajıdır: **yorumlanabilirlik.**

<br>

---

<br>

### 🏘️ Bellek Tabanlı Sınıflandırma (KNN)

KNN (K-En Yakın Komşu), entropiye dayalı algoritmaların aksine **kural üretmez**. Bunun yerine eğitim verisini bellekte tutar ve yeni bir örnek geldiğinde **en benzer K komşuyu** bularak çoğunluk oylamasıyla sınıf atar.

```
Yeni müşteri geldi → Tüm eğitim verisiyle mesafe hesapla
                   → En yakın K=14 komşuyu seç
                   → Çoğunluk hangi sınıftaysa → o sınıf
```

<table>
<tr>
<td>✅ <b>Avantaj:</b> Eğitim süresi çok kısa, yeni veriye kolayca uyum sağlar</td>
</tr>
<tr>
<td>❌ <b>Dezavantaj:</b> Tahmin süresi uzun (tüm veriyle mesafe hesaplar), ölçeklendirme şart</td>
</tr>
</table>

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

## ⚙️ Hiperparametre Seçimleri

<br>

### 🌳 Karar Ağacı — GridSearchCV ile Optimize Edildi

> 168 farklı parametre kombinasyonu × 5 fold = **840 model eğitildi**, en iyisi seçildi.

| Parametre | Değer | Açıklama |
|:---|:---:|:---|
| `max_depth` | **3** | Ağaç en fazla 3 seviye derine iner — fazla derin = ezberleme (overfitting) |
| `criterion` | **gini** | Dallanma kararı Gini impurity ile yapılır (entropy de denendi) |
| `min_samples_split` | **2** | Bir düğümün bölünmesi için en az 2 örnek gerekli |
| `min_samples_leaf` | **10** | Her yaprak düğümde en az 10 örnek olmalı — aşırı öğrenmeyi engeller |
| `class_weight` | **balanced** | Azınlık sınıfına (ayrılan) daha fazla ağırlık verir |

<details>
<summary><b>🔍 GridSearchCV nedir?</b> (tıkla)</summary>
<br>
Tüm parametre kombinasyonlarını dener, her birini 5 kez farklı veri dilimleriyle test eder ve en iyi sonucu verenini otomatik seçer. Bu projede <code>max_depth</code> (7 değer) × <code>criterion</code> (2 değer) × <code>min_samples_split</code> (4 değer) × <code>min_samples_leaf</code> (3 değer) = 168 kombinasyon × 5 fold = <b>840 model</b> eğitildi.
</details>

<details>
<summary><b>🔍 Gini vs Entropy — fark ne?</b> (tıkla)</summary>
<br>

Her ikisi de dallanma noktasında saflığı ölçer:

- **Entropy:** H = −Σ pᵢ · log₂(pᵢ) → ID3/C4.5 algoritmalarının kullandığı bilgi kazancı ölçüsü
- **Gini:** G = 1 − Σ pᵢ² → CART algoritmasının kullandığı ölçü, hesaplama olarak daha hızlı

Pratikte sonuçlar genellikle çok benzer çıkar. Bu projede GridSearchCV her ikisini de denedi ve **gini** daha iyi sonuç verdi.

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

Karar ağacı hangi bilgilere bakarak karar veriyor? (Information Gain / Gini Importance)

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
