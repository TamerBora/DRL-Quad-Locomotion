# Guided SAC: Eleştirel Değerlendirme ve Geliştirme Yol Haritası

**Kapsam:** Haklıdır & Temeltaş (2021) Guided SAC ve Çerçi & Temeltaş (2024) Multi-Agent Guided DRL yöntemlerinin, tekerlekli-bacaklı quadruped lokomosyon bağlamında (TÜBİTAK 1001 – 125E330) SAC ve PPO'ya karşı rekabetçi bir öğrenme çerçevesine dönüştürülmesi.

**Hedef iddia:** "Guided SAC, kısmi gözlem altında PPO'dan daha yüksek asimptotik performansa, SAC'tan daha iyi örnek verimliliğine ve her ikisinden daha iyi sim-to-real transferine ulaşır — ve bu kazanç ablation ile mekanizmalara atfedilebilir."

Bu doküman, o iddianın hangi koşullarda savunulabilir olduğunu ve hangi koşullarda çürüyeceğini ortaya koymak üzere yazılmıştır.

---

## 1. Yöntemin anatomisi: üç ayrı mekanizma

Guided SAC tek bir fikir değil, birbirine karışmış üç mekanizmadır. Literatürde ve her iki makalede bunlar hiç ayrıştırılmamıştır. Geliştirmenin ilk adımı bu ayrıştırmadır.

### M1 — Asimetrik critic

`Q(s, a)` tam durumu görür, `π(a|o)` görmez. Privileged bilgi actor'a yalnızca `∇_a Q` üzerinden sızar.

- **Maliyet:** Sıfır ek ağ. Yalnızca critic girdi boyutu artar.
- **Kaynak:** Pinto et al. 2017 (Asymmetric Actor Critic). Guided SAC'a özgü değil.
- **Beklenen katkı:** Yüksek. Kısmi gözlemle eğitilen bir critic, perceptual aliasing nedeniyle aynı `o`'ya karşılık gelen farklı `s`'lerin değerlerini ortalar; bu bias tüm policy gradient'e sızar.

### M2 — Guiding actor'ın behavior policy olarak kullanılması

Eq. (11): `D = (a₀ᵍ, s₀ᵍ, r₀ᵍ, a₁ᶠ, s₁ᶠ, r₁ᶠ, a₂ᵍ, …)` — iki aktör aynı ortamda dönüşümlü aksiyon üretir.

- **Maliyet:** Bir ek aktör ağı (forward + backward).
- **Beklenen katkı:** Buffer'ın state-visitation support'unu genişletir. M1'in **sağlayamayacağı** tek şey budur: privileged critic, mevcut veri üzerinde credit assignment'ı düzeltir; yeni veri üretmez.
- **Koşul:** Ancak exploration gerçekten darboğazsa değerlidir. Shaped reward + geniş domain randomization + 4096 paralel ortam olan locomotion setup'ında exploration çoğu zaman darboğaz **değildir**. Bu, M2'nin locomotion'da MuJoCo POMDP task'larındakinden daha az işe yarayacağı anlamına gelir.

### M3 — Aksiyon uzayında imitasyon

Çerçi & Temeltaş (2024), Eq. (18) ve TÜBİTAK formu, İP-3:

>  **L(π)  =  E_o [ α · log π(a|o)  −  Q(s,a)  +  λ · ‖a − a_g‖² ]**

- **Maliyet:** Sıfıra yakın (bir norm terimi).
- **Beklenen katkı:** Aksiyon uzayında yoğun, düşük varyanslı, bootstrapping'den bağımsız supervised hedef. Erken eğitimde `Q` gürültülüyken bile anlamlı gradyan taşır. Ayrıca trust-region/anchor etkisiyle güncelleme varyansını düşürür.
- **Risk:** Bölüm 2.1'deki imitation gap. Bu, yöntemin en ciddi teorik zayıflığıdır.

> **Ana hipotez (test edilmeli):** Rapor edilen kazancın büyük kısmı M1'den gelir. M1 bedavadır; M2 pahalıdır. Eğer bu doğruysa, çok etmenli mimaride her etmene guiding actor eklemek (4× aktör) gerekçesiz kalır. Bunu İP-3 tasarımı dondurulmadan önce bilmek gerekir.

---

## 2. Kritik zayıflıklar

### 2.1 Imitation gap — en ciddi problem

`π_g`, tam gözlemli MDP'nin çözümüdür. `π_f`, POMDP'nin çözümü olmalıdır. Bunlar farklı problemlerin çözümleridir ve birini diğerine projekte etmek genel olarak **optimal değildir**. Warrington et al. 2021 (*Robust Asymmetric Learning in POMDPs*) bunu formalize eder: teacher'a imitasyon zorlaması, elde edilebilecek en iyi kör politikadan **daha kötü** bir sonuca yakınsayabilir.

**Makale kendisiyle çelişiyor.** Haklıdır & Temeltaş (2021) girişi motivasyonunu şuna dayandırır:

> *"…the agent must consider information gathering and therefore engage in much more exploration and interaction with these environments to learn how to collect and use the information."*

Ancak tam durumu gören bir teacher, tanımı gereği bilgi toplama davranışı **üretmez** — buna ihtiyacı yoktur. Dolayısıyla `a^g`'ye çekmek, motivasyon olarak gösterilen davranışı bastırır.

**Bu sizin senaryonuzda somuttur.** Blind proprioceptive locomotion'da robot ayağıyla zemini yoklar (foot probing), gövde açısını değiştirerek IMU sinyalini zenginleştirir, temkinli adım atar. Height scan gören bir teacher bunların hiçbirini yapmaz. Aksiyon uzayında imitasyon, tam olarak deploy'da hayatta kalmayı sağlayan davranışları cezalandırır.

### 2.2 Off-policy tutarlılığı

Buffer artık iki farklı politikadan aksiyon içerir. Makale bunu importance sampling düzeltmesiyle değil, "`α` ikisini aynı politikaya yakınsatır" argümanıyla geçiştirir (2021, Bölüm IV). Bu bir düzeltme değil, temennidir. Distribution shift `λ` büyüdükçe ve iki politikanın gözlem asimetrisi arttıkça büyür.

### 2.3 Deneysel zayıflıklar

- **Ablation yok.** M1/M2/M3 hiç ayrıştırılmamış.
- **Etki büyüklüğü varyansın içinde.** Tablo 3'te 7–20% kazanç raporlanıyor ama Şekil 5'te Hopper-v2 ve Walker2d-v2 POMDP eğrilerinde confidence band'ler tamamen iç içe. InvertedDoublePendulum'da TD3 kazanıyor (9306 vs 9120). 10 seed iyi bir uygulama, ama etki büyüklüğü onu haklı çıkarmıyor.
- **POMDP kurgusu yapay.** Gözlem vektörünün sabit boyutlarını sıfırlamak (Tablo 5) gerçek kısmi gözlem değildir — maskenin hangi boyutlarda olduğu sabit ve öğrenilebilir. Sizin blind locomotion senaryonuz çok daha dürüst bir POMDP'dir ve bu **avantajınızdır**, dezavantajınız değil.
- **Notasyon hataları.** Algorithm 1 satır 4'te iki politika da `π_θg` yazılmış. Eq. (14) ve (15) `θ_f`/`θ_g` dışında birebir aynı — ki bu yanlış: guiding actor `s`, control actor `h` görmeli. Yeniden implementasyonda makaleyi değil kendi türetmenizi baz alın.

### 2.4 SAC'ın locomotion'daki yapısal dezavantajı (guided'dan bağımsız)

Guided katmanı tartışılmadan önce kabul edilmesi gereken bir gerçek: **massively parallel simülasyonda PPO'nun üstünlüğü algoritmik değil, sistemseldir.**

| | PPO | SAC |
|---|---|---|
| Env throughput kullanımı | 4096 env'i doyurur, rollout → update senkron | Replay buffer, GPU'da veri hareketi darboğazı |
| Wall-clock / iterasyon | Düşük | Yüksek (UTD ratio × gradient step) |
| Domain randomization altında | On-policy, veri her zaman taze | Buffer'daki eski veri farklı DR parametrelerinden — off-policy + non-stationary dinamik |
| Replay buffer belleği | Yok | 1M × (obs + priv_obs) — 187-boyutlu height scan ile ciddi |

Sizin daha önce tespit ettiğiniz **actuator-gain DR'ın multimodal Q-landscape yaratması** hipotezi tam bu tabloya oturuyor: DR parametreleri buffer boyunca değiştiği için `Q(s,a)` aslında farklı MDP'lerin değer fonksiyonlarının bir karışımını öğrenmeye zorlanıyor. Bu, rough terrain SAC koşusunun standing crouch'a çökmesinin en olası açıklamasıdır — crouch, tüm DR modları altında ortalama olarak "güvenli" olan tek davranıştır.

**Sonuç:** Guided SAC'ı PPO'dan üstün kılmak istiyorsanız, guided katmanını iyileştirmek yeterli değildir. SAC'ın DR altındaki off-policy patolojisini de çözmek zorundasınız. Yol haritasının Faz 3'ü buna ayrılmıştır.

---

## 3. Yol haritası

### Faz 0 — Zemin hazırlığı (2–3 hafta)

**Amaç:** Ölçülebilir bir taban ve dürüst bir karşılaştırma protokolü.

1. **Referans implementasyon.** ETH RSL-RL-SAC (arXiv:2605.24975) üzerine kurun. Makaleyi yeniden implemente etmeyin; SpinningUp tabanlı 2021 kodu IsaacLab ölçeğine uygun değil.
2. **Karşılaştırma protokolü — iki eksende raporlayın:**
   - Örnek verimliliği: return vs. environment steps
   - Hesaplama verimliliği: return vs. wall-clock (aynı donanım)
   
   PPO'yu yalnızca birinci eksende yenmek yayın için yetersizdir; ikinci eksende kaybediyorsanız bunu açıkça yazın. Tek eksende raporlamak literatürdeki en yaygın SAC-vs-PPO yanılsamasıdır.
3. **Seed protokolü.** En az 5 seed, IQM + stratified bootstrap CI (Agarwal et al. 2021, *Deep RL at the Edge of the Statistical Precipice*). Mean ± std kullanmayın — 2021 makalesinin Şekil 5'i tam bu yüzden ikna edici değil.
4. **Privileged/proprioceptive ayrımını dondurun ve belgeleyin.**

   | Kanal | İçerik |
   |---|---|
   | `o_t` (control actor) | Eklem pozisyon/hız, IMU (açısal hız, projected gravity), önceki aksiyon, hız komutu, tekerlek hızları — **geçmiş penceresi K** |
   | `s_t` (privileged) | `o_t` + gövde çizgisel hızı, height scan (187), sürtünme katsayısı, temas kuvvetleri, gövde kütle/COM offset, actuator gain DR parametreleri |

   Kritik ayrıntı: **DR parametrelerini privileged kanala koyun.** Bu, 2.4'teki multimodal Q problemini kısmen çözer — critic artık hangi MDP'de olduğunu bilir.

**Kill criterion:** Faz 0 sonunda vanilla SAC, aynı wall-clock bütçesinde PPO'nun %60'ına ulaşamıyorsa, SAC'ın kendisi ile ilgili bir problem var demektir; Faz 1'e geçmeden Faz 3'ün ilk maddelerini uygulayın.

---

### Faz 1 — Ablation merdiveni (4–6 hafta) — **Projenin en yüksek getirili kısmı**

Artan maliyet sırasıyla, hepsi aynı bütçede:

| # | Konfigürasyon | İzole ettiği şey |
|---|---|---|
| A1 | Vanilla SAC: `Q(o,a)`, `π(a|o)` | Taban |
| A2 | Asimetrik critic: `Q(s,a)`, `π(a|o)` | **M1** |
| A3 | A2 + latent supervision: `π(a|o,ẑ)`, `L=‖ẑ−z_priv‖²` | HIM/RMA tarzı alternatif |
| A4 | A2 + guiding actor **sadece behavior policy** (λ=0) | **M2** |
| A5 | A4 + imitasyon (λ>0) = tam Guided SAC | **M3** |
| A6 | PPO + aynı asimetrik yapı | Algoritma sınıfı etkisi |

**Bu tablo tek başına yayınlanabilir bir katkıdır.** Guided SAC literatüründe hiç yapılmamıştır ve sonucu ne olursa olsun bilgi değeri taşır.

**Beklenti (dürüstçe):** Kazancın büyük kısmının A1→A2 adımında oluşması yüksek olasılıklıdır. A4→A5'in negatif çıkması da ciddi bir olasılıktır (imitation gap). Her iki sonuç da yayınlanabilir; ikincisi daha ilginçtir.

**Karar noktası:** A5 ≤ A2 ise, ikinci aktör mevcut haliyle savunulamaz. Faz 2'ye geçin — orada ikinci aktörü kurtarma stratejileri var. Faz 2 de işe yaramazsa, projeyi A3 (latent supervision) hattına kaydırın ve bunu bir bulgu olarak raporlayın.

---

### Faz 2 — Imitation gap'i kapatma (6–8 hafta)

Sırayla artan iddia gücüyle dört strateji:

**2.1 λ curriculum (en ucuz, hemen yapın)**

Mevcut formülasyonda `λ` sabit. Ama teacher yakınsamadan önce `a^g` gürültüdür. Öneri:

>  **λ(t)  =  λ_max · σ( ( J(π_g) − J_esik ) / τ )**

veya daha basiti: teacher'ın TD-error'una ters orantılı. Ek olarak eğitim sonuna doğru `λ → 0` (annealing) — student son aşamada kendi kısıtları altında serbest kalmalı.

**2.2 Aksiyon uzayı yerine latent uzay denetimi (en yüksek getiri)**

Aksiyon uzayında imitasyon, "şu aksiyonu ver" der. Latent uzayda denetim, "çevre şuydu" der ve politikayı bu bilgiyi kendi kısıtları altında nasıl kullanacağı konusunda serbest bırakır. Imitation gap tanım gereği kapanır.

>  **z_hat(t)  =  E_ψ( o(t−K:t) )**
>
>  **L_enc  =  ‖ z_hat(t) − z_priv(t) ‖²**
>
>  **π( a | o(t), z_hat(t) )**

Bu, projenin ikinci makalesindeki (Çerçi) yaklaşımı ile HIM estimator'ı birleştirir ve TÜBİTAK formundaki Dayanıklı Adaptasyon Modülü ile **doğal olarak aynı yapıdır** (form Bölüm 2.4'teki epistemik ağ, `z_t = φ(h_t, ξ)`). Yani form zaten bu yönü içeriyor; yapılması gereken, DAM'ı bir "eklenti" olarak değil, guiding actor'ın **yerine geçen** ana mekanizma olarak konumlandırmak.

**2.3 Hibrit: yumuşak imitasyon + latent denetim**

`λ‖a−a^g‖²` yerine KL bazlı, asimetrik bir terim:

>  **L_imit  =  λ · D_KL( π_f(· | o, z_hat)  ‖  stopgrad[ π_g(· | s) ] )**

L2 yerine KL kullanmanın avantajı: SAC politikaları zaten Gaussian, dolayısıyla KL kapalı formda ve **varyansı da eşleştirir**. L2 sadece ortalamayı çeker — teacher'ın belirsiz olduğu durumlarda student'ı sahte bir kesinliğe zorlar. Bu ince ama önemli bir fark.

**2.4 Adaptive teacher (en iddialı)**

Warrington et al.'un asıl önerisi: teacher'ı sabit tutmayın, **student'ın kısıtlarına duyarlı hale getirin**. Teacher'ın hedefi "MDP'yi optimal çöz" değil, "student'ın taklit edebileceği en iyi davranışı üret" olur:

>  **max over π_g :   J(π_g)  −  β · E[ D_KL( π_f(· | o) ‖ π_g(· | s) ) ]**

Bu, teacher'ı gözlemlenebilir bilgiye dayanan davranışlara doğru iter ve foot-probing gibi information-gathering davranışlarının bastırılmasını engeller. Uygulaması zor ama gerçek bir bilimsel katkıdır.

---

### Faz 3 — SAC'ı DR altında sağlamlaştırma (6–8 hafta) — **PPO'yu yenmek için zorunlu**

Faz 1–2 guided katmanını iyileştirir. Bu faz, PPO ile rekabetin asıl koşuludur.

**3.1 Multimodal Q problemi (sizin çökme hipoteziniz)**

- **DR parametrelerini privileged critic girdisine ekleyin** (Faz 0.4). Q artık koşullu: `Q(s, ξ_DR, a)`. Bu, farklı MDP'lerin karışımını öğrenme baskısını kaldırır.
- **Distributional critic** (C51/QR-DQN'in sürekli muadili, ya da basitçe quantile Huber loss). Multimodal getiri dağılımını ortalamaya çökertmez.
- **Dağılımı daraltarak başlayın:** DR curriculum. Dar DR ile başlayıp genişletin. Bu, standing-crouch attractor'ından kaçmanın en pratik yolu.

**3.2 Modern SAC stabilizasyon teknikleri**

Bunlar 2021 makalesinden sonra çıktı ve etkileri büyük:

| Teknik | Ne yapar | Kaynak |
|---|---|---|
| **Layer normalization** critic'te | Yüksek UTD altında değer patlamasını önler | Nauman et al. 2024 (BRO) |
| **CrossQ** | Target network'ü kaldırır, BatchRenorm kullanır; UTD=1'de SAC'ın UTD=20 performansı | Bhatt et al. 2024 |
| **Critic ensemble (N>2) + pessimism** | Overestimation'ı DR altında kontrol eder | REDQ, DroQ |
| **n-step return (n=3–5)** | Bootstrapping hatasını azaltır, locomotion'da belirgin | — |
| **UTD ratio taraması** | SAC'ın örnek verimliliği buradan gelir; paralel env'de UTD tanımı dikkat ister | REDQ |

**Paralel ortamda UTD tanımı:** 4096 env × 1 step = 4096 transition/iterasyon. UTD'yi "gradient step / env step" olarak tanımlarsanız SAC'ın klasik UTD=1 ayarı burada anlamsız derecede düşük kalır. Bu, IsaacLab'de SAC'ın "çalışmamasının" yaygın ve gözden kaçan sebebidir. Env sayısını düşürüp (256–512) UTD'yi yükseltmek, 4096 env + düşük UTD'den genellikle daha iyidir.

**3.3 Replay buffer tasarımı**

- Height scan'i (187 boyut) buffer'da `float16` saklayın veya hiç saklamayın — her sample'da simülasyondan yeniden hesaplamak bazen daha ucuzdur.
- Prioritized replay'i DR ile birlikte dikkatli kullanın: öncelik, zor DR modlarına aşırı odaklanmaya yol açabilir.

---

### Faz 4 — Dürüst karşılaştırma ve iddia oluşturma (3–4 hafta)

Aşağıdaki tabloyu doldurun. Boş bırakılan hücre yoksa yayın hazırdır.

| Metrik | PPO | SAC | Guided SAC (final) |
|---|---|---|---|
| Return @ 100M env steps | | | |
| Return @ 4 saat wall-clock | | | |
| Env steps to 80% asymptote | | | |
| Rough terrain success rate | | | |
| Cost of Transport | | | |
| OOD terrain (eğitimde görülmeyen) | | | |
| Sim-to-real transfer gap | | | |
| Seed varyansı (IQM CI genişliği) | | | |

**Beklenen dürüst sonuç:** Guided SAC örnek verimliliğinde ve OOD genellemede kazanır; wall-clock'ta PPO'ya kaybeder veya başabaş kalır. Bu **iyi bir sonuçtur** ve şöyle konumlandırılır:

> *"Gerçek robotta veri toplamanın pahalı olduğu veya simülasyonun yavaş olduğu rejimlerde (yüksek fideliteli temas modelleri, deformable terrain, gerçek robot fine-tuning), örnek verimliliği wall-clock'tan daha önemli hale gelir. Guided SAC bu rejimi hedefler."*

Bu konumlandırma savunulabilir. "Her açıdan PPO'dan iyidir" konumlandırması savunulamaz ve reviewer tarafından hemen yıkılır.

---

### Faz 5 — Çok etmenli ölçekleme (TÜBİTAK İP-3, 8–10 hafta)

**Faz 1'in sonucunu beklemeden başlamayın.** İP-3'ün mevcut tasarımı her bacağa ayrı guiding actor öngörüyor (4× aktör). Eğer A5 ≤ A2 çıkarsa bu tasarım gerekçesiz olur.

Ölçekleme sırasında dikkat:

1. **Parameter sharing zorunlu.** Form bunu içeriyor, doğru tercih. 4 bağımsız aktör hem hesaplama hem örnek verimliliği açısından savunulamaz. Bacak kimliğini one-hot veya öğrenilen embedding ile verin.
2. **Tek guiding actor, paylaşımlı.** Her etmene ayrı teacher gereksiz — teacher zaten tam durumu görür, dolayısıyla merkezi olabilir. `π_g(a^{1:4} | s)` tek ağ, `π_f^i(a^i | o^i)` paylaşımlı.
3. **Credit assignment.** Ortak ödül fonksiyonu ile 4 etmen → form Risk Tablosu 3'te bu risk doğru tespit edilmiş. Çözüm olarak "yerel ödül bileşenleri" öneriliyor; buna ek olarak **counterfactual baseline** (COMA tarzı) veya value decomposition (QMIX'in sürekli muadili) düşünülmeli.
4. **Gerçek soru:** Bacak-başına-etmen ayrıştırması lokomosyonda *gerçekten* kazandırıyor mu? Quadruped lokomosyonda bacaklar arası bağımlılık çok yüksektir (gövde tek bir rigid body). Tek etmenli merkezi politikaya karşı bir ablation şart — aksi halde çok etmenli yapı, ödenmiş ama getirisi ölçülmemiş bir karmaşıklıktır.

---

### Faz 6 — Sim-to-real (TÜBİTAK İP-5)

- **Guided yapının asıl vaadi burada test edilir.** Sim-to-real gap, tanım gereği bir OOD problemidir; asimetrik eğitimin değeri en çok burada görünmelidir. Faz 4 tablosunun son iki satırı bu fazın çıktısıdır.
- **Deploy maliyeti sıfırdır** — guiding actor atılır, çalışan ağ vanilla SAC ile birebir aynıdır. Bunu makalede açıkça belirtin; asimetrik yöntemlerin en güçlü satış argümanıdır.
- **Fail-safe** (form İP-4 risk tablosunda var): epistemik belirsizlik eşiği aşıldığında kontrollü duruş. Bunu Go2-W üzerinde gerçekten uygulayın, formdaki bir cümle olarak bırakmayın.

---

## 4. Karar noktaları özeti

| Nokta | Koşul | Aksiyon |
|---|---|---|
| Faz 0 sonu | SAC, PPO'nun %60 wall-clock'una ulaşamıyor | Faz 3'ün 3.1–3.2'sini öne al |
| Faz 1 sonu | A5 ≤ A2 | İkinci aktörü mevcut haliyle terk et → Faz 2 |
| Faz 2 sonu | Hiçbir varyant A2/A3'ü geçmiyor | Latent supervision (A3) hattına geç, bulguyu negatif sonuç olarak yayınla |
| Faz 4 sonu | Guided, hiçbir eksende kazanmıyor | Katkıyı "kapsamlı ablation + negatif sonuç" olarak konumlandır — bu da yayınlanabilir |
| Faz 5 öncesi | Çok etmenli ≤ tek etmenli merkezi | İP-3'ün tasarımını yeniden gözden geçir |

---

## 5. Riskler

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| M2/M3'ün katkısı ölçülemeyecek kadar küçük | **Yüksek** | Projenin ana özgünlük iddiası zayıflar | Faz 1'i erken yap; negatif sonuç yayın stratejisini hazır tut |
| SAC, DR altında IsaacLab'de PPO'ya wall-clock'ta hiç yaklaşamaz | Orta-yüksek | Karşılaştırma tek eksene sıkışır | Faz 4'teki "pahalı veri rejimi" konumlandırması |
| Imitation gap, blind locomotion'da özellikle şiddetli | Orta | Guided SAC vanilla SAC'ın altına düşer | λ annealing + latent supervision'a geçiş |
| Çok etmenli ayrıştırma lokomosyonda kazandırmıyor | Orta | İP-3'ün %30 ağırlığı riske girer | Tek etmenli merkezi ablation'ı İP-3 içine yaz |
| Buffer belleği (height scan) donanımı aşar | Düşük-orta | Eğitim ölçeklenemez | float16 / yeniden hesaplama |

---

## 6. Yayın stratejisi

| Çıktı | İçerik | Hedef |
|---|---|---|
| **P1** | Guided SAC'ın mekanizma ablation'ı (Faz 1) + PPO/SAC dürüst karşılaştırma | Konferans (IROS/ICRA/CoRL) — tek başına yeterli |
| **P2** | Imitation gap'in blind locomotion'da ölçümü + latent supervision ile çözümü (Faz 2) | Q1 dergi (RA-L / T-RO) |
| **P3** | Çok etmenli ölçekleme + DAM + sim-to-real (Faz 5–6) | TÜBİTAK ana çıktısı, Q1 dergi |

P1'in en hızlı ve en düşük riskli çıktı olduğunu vurgulamak gerekir: sonucu ne olursa olsun yayınlanabilir, çünkü literatürdeki gerçek bir boşluğu doldurur.

---

## 7. Bu dokümanın temel mesajı

Guided SAC'ın literatürdeki mevcut hali, üç mekanizmanın ayrıştırılmamış bir paketidir ve bu paketin en pahalı bileşeni (ikinci aktör) muhtemelen en az katkıyı sağlamaktadır. En ucuz bileşeni (asimetrik critic) ise Guided SAC'a özgü değildir.

Yöntemi gerçekten ileri taşımanın yolu, ikinci aktörü savunmaktan değil, **ne zaman ve neden gerekli olduğunu ölçmekten** geçer. Ölçüm negatif çıkarsa, latent supervision hattı (form Bölüm 2.4'te zaten var) daha savunulabilir bir yapıya işaret ediyor — ve TÜBİTAK projesinin özgün değer iddiası bu hatta da aynı derecede güçlü kalır.
