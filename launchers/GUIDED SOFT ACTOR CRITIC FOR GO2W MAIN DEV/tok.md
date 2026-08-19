# TOK 2026 — Go2-W üzerinde PPO ve SAC: devir teslim notu

Son güncelleme: 2026-08-15. Bu belge bildiriyi ve SAC kolunun FlashSAC'ten nasıl
kurulduğunu anlatır. Amaç, işi devralan birinin kaynak koda inmeden neyin nasıl
yapıldığını, hangi sayının nereden geldiğini ve nelerin doğrulanmış olduğunu
görebilmesidir.

Yol adları `quadruped_lab-tamer-env/` köküne göredir.

---

## 1. Bildiri

| | |
|---|---|
| Kaynak | `TOK2026Conference/root.tex` (tek dosya, 730+ satır) |
| Şablon | `TOKsty.sty` + `easyturk.sty` + `IEEEbib` — 4–6 sayfa, iki sütun |
| Durum | 6 sayfa, overfull kutu yok, kırık gönderme yok, 16 künye, yer tutucu yok |
| Şekiller | `TOK2026Conference/figures/` — `make_figs.py` üretir |
| Derleme | Yazar pdfLaTeX kullanıyor. Yerelde tectonic (`conda activate tex`) var ama **XeTeX koşuyor ve Türkçe ı/ş/ğ karakterlerini düşürüyor** — sayfa ölçümleri gösterge niteliğinde, kesin değil |

**Ana bulgu:** 50M ortam adımlık eşit bütçede, engebeli arazi müfredatı altında
iki yöntem de yanal harekete ulaşamıyor. Görev düz zemine indirgendiğinde
**yalnızca FlashSAC** yanal hareketi kazanıyor (0,416 m/s, komutun %83'ü); aynı
koşulda eğitilen PPO 0,0006 m/s'de kalıyor. PPO bu düzeye ancak ~30 kat veriyle
(1,475×10⁹ adım) ulaşıyor.

---

## 2. SAC kolu: FlashSAC nasıl kuruldu

### 2.1 Vendored kaynak ve köken

```
flashsac_go2w_fable/
├── FlashSAC/flash_rl/       # upstream, algoritma çekirdeği BAYT BAYT AYNI
├── go2w_tasks/              # görev kaydı (yalnızca sighted ablasyonu)
├── hparams.py               # TEK DOĞRULUK KAYNAĞI — train.py defaults buradan
├── train.py                 # eğitim betiği
├── action_bounds.py         # eylem sınırı kalibrasyonu
├── calibration.json         # PPO eylem yüzdelikleri (185 dosya, 2,8e8 örnek)
├── compare_params.py        # adillik tablosunu üretir (drift koruması)
├── export_policy_go2w.py    # TorchScript dışa aktarım (sim2sim/donanım)
└── VENDORED.md              # köken + her sapmanın gerekçesi
```

Upstream: `github.com/Holiday-Robot/FlashSAC`, commit `87edc90`, MIT.
`VENDORED.md`'ye göre **öğrenme kuralına dokunulmadı**; tüm sapmalar
`flash_rl/envs/isaaclab.py` sarmalayıcısında ve entegrasyon kaynaklı.

### 2.2 FlashSAC vanilla SAC değil — bunu bilmek şart

Algoritma çekirdeğini okudum, SAC amacının üstüne şunları ekliyor:

| bileşen | nerede | vanilla SAC'te |
|---|---|---|
| dağılımsal (kategorik) kritik | `flashSAC/update.py:_compute_categorical_td_target` | yok, skaler Q |
| $n$ adımlı getiri, $n=3$ | `hparams.py:33` | yok, tek adım |
| `UnitRMSNorm` / `EnsembleUnitRMSNorm` | `flashSAC/network.py` | yok |
| ödül normalizasyonu | `flashSAC/agent.py:RewardNormalizer` | yok |

Upstream özeti bunu şöyle özetliyor: *"sharply reduces gradient updates while
compensating with larger models and higher data throughput"* + *"explicitly
bounds weight, feature, and gradient norms."*

Bildiride bu, Tablo 1'e satır olarak yazıldı (`Kritik / dağılımsal`,
`Getiri / 3 adımlı`, `Normalizasyon / RMS, ödül`) — gövde metninde ilan
edilmedi, yazarın tercihi.

### 2.3 Gözlem sözleşmesi — PPO ile eşleştirmenin özü

Tek yerde tanımlı: `source/quadruped_lab/.../unitree_go2w/rough_env_cfg.py:94-95`
aktör grubundan `base_lin_vel` ve `height_scan`'i düşürüyor, kritik grubu ikisini
de görüyor. **İki kol da aynı ortamdan aynı iki grubu alıyor.**

- Aktör: 57 boyutlu çerçevenin 5 zaman adımlık geçmişi = **285**, dış algılayıcısız
- Kritik: 60 + 187 yükseklik taraması = **247**, ayrıcalıklı

`hparams.py:46-47` bunu sabitliyor (`ACTOR_OBS`, `CRITIC_OBS`);
`tests/smoke_env.py` canlı ortama karşı doğruluyor. PPO tarafında `agent.yaml`'da
`critic_hidden_dims` var ve rsl_rl'in `default_sets = ["critic"]` varsayılanı
ayrıcalıklı grubu alıyor.

### 2.4 Eylem sınırı kalibrasyonu

SAC'nin tanh çıkışı [-1,1] ile sınırlı; ölçek PPO'nun **kendi** eylem
yüzdeliklerinden türetildi (`calibration.json`, p99,9):

- bacaklar **5,0** (p99,9 = 5,13)
- tekerlekler **10,0** (p99,9 = 10,07)

Bu algoritmik değişiklik değil, sınırlı/sınırsız eylem uzaylarını eşleştiren
kalibrasyon. Yapılmazsa SAC en yüksek komut hızına yapısal olarak ulaşamaz.
(Sabatini ve ark. bunu bağımsız olarak kararsızlığın *birincil* nedeni sayıyor.)

### 2.5 Hiperparametreler: eşleşen ve sapan

`hparams.py` tek doğruluk kaynağı. Referans = FlashSAC'in resmi
`scripts/run_isaaclab.sh`.

| | referans | bizim | not |
|---|---|---|---|
| `num_envs` | 1024 | **256** | SAPMA — 8 GB VRAM |
| `buffer_size` | 10M | **2M** | SAPMA — 30 GB host RAM |
| `buffer_device` | cuda | **cpu** | SAPMA — pinned host RAM, VRAM'e dokunmaz |
| `updates_per_interaction_step` | 2,0 | **0,5** | **SAPMA DEĞİL**: 0,5/256 = 2/1024, UTD eşleşti |
| `total_steps` | — | 50.000.896 | PPO ile eşit bütçe |
| `batch_size` | — | 2048 | |
| `gamma` | — | 0,99 | PPO ile eşleşti |
| `n_step` | — | 3 | |

UTD = 1,95×10⁻³ (= 0,5/256). Bu **düşük ama handikap değil** — FlashSAC'in
tasarımı büyük ağ + büyük toplu iş + seyrek güncelleme. Bildiride Sınırlar (iii)
böyle yazılı. Buna karşılık `num_envs` gerçekten referansın altında, o yüzden
ölçülen üstünlük **alt sınır** olarak sunuluyor.

### 2.6 Eğitim komutları

Görev kimlikleri:
- Engebeli (ana): `QuadrupedLab-Isaac-Velocity-Rough-Unitree-Go2W-Fable-v0`
- Düz: `QuadrupedLab-Isaac-Velocity-Rough-Unitree-Go2W-Fable-FlatFT-v0`
  (alt araziler düz zeminle değiştirilmiş; ödül, DR ve gözlem grupları AYNI)

```bash
cd /home/tamer/robotics/launchers/quadruped_lab-tamer-env/flashsac_go2w_fable
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab

python train.py --arm blind --seed 0 --headless \
  --task QuadrupedLab-Isaac-Velocity-Rough-Unitree-Go2W-Fable-v0 \
  --num_envs 256 --total_steps 50000896 \
  --updates_per_interaction_step 0.5 --batch_size 2048 \
  --buffer_size 2000000 --buffer_min 50000 --buffer_device cpu \
  --gamma 0.99 --n_step 3 \
  --leg_action_bound 5.0 --wheel_action_bound 10.0 \
  --terrain_exposure curriculum --reward_variant fable \
  --save_every_env_steps 2500000 \
  --logger wandb --wandb_mode online \
  --project_name tok2026_go2w_ppo_vs_sac --exp_name flashsac_blind_s0
```

**`--ml_framework jax` KULLANMAYIN** — FlashSAC PyTorch. O bayrak yalnızca
SBX/JAX boru hatları için.

Elektrik kesintisi/uyku CUDA bağlamını öldürüyor; uzun koşularda
`systemd-inhibit --what=sleep:idle` ile sarın. tmux/nohup kullanılmıyor.

---

## 3. Raporlanan denetim noktaları

| kol | denetim noktası | not |
|---|---|---|
| FlashSAC engebeli 50M | `flashsac_go2w_fable/logs/flashsac_blind_s0/step166005` | 42,5M adım, tepe |
| FlashSAC düz 50M | `eval_suite/results_tok/models/flashsac_flat_s0_step195310` | 50M, son (çökme yok) |
| FlashSAC 200M (doygunluk) | `logs/flashsac_blind_200M_s0_r1/step410151` | ayrı koşu |
| PPO engebeli 50M | `logs/rsl_rl/unitree_go2w_rough_fable/2026-08-07_12-48-55/model_508.pt` | son yineleme |
| PPO düz 50M | `.../2026-08-09_19-45-38/model_508.pt` | yalnızca kontrol |
| PPO üst referans | `ros2_ws/.../ppo_fable_rough/model_15000.pt` | 1,475×10⁹ adım |

Seçim protokolü: adaylar **ayrık tohumlarda (5,6,7)** tarandı, seçilen nokta
**raporlama tohumlarında (0–4)** ölçüldü. `flashsac_flat_s0_step195310/`
içinde `PROVENANCE.txt` var — wandb koşu adı, bütçe, config, seçim gerekçesi.

---

## 4. Değerlendirme

```bash
# 19 senaryo: flat/rough_easy/mid/hard (vx sweep) + slope_05..30 + stair_05..25
CKPT_GO2W=<yol> SEEDS=0,1,2,3,4 NUM_ENVS=128 OUT_DIR=eval_suite/results_tok/<ad> \
  bash eval_suite/run_matrix.sh go2w
# iniş + çok yönlü (omni_lat, omni_yaw)
CKPT_GO2W=<yol> SEEDS=... OUT_DIR=... bash eval_suite/run_matrix_extension.sh go2w
```

Toplam 21 senaryo, 37 (senaryo, komut) hücresi, kol başına 23.680 bölüm.

**Başarı ölçütü** (`eval_suite/metrics.py:142-159`) — bildiride yıllarca yanlış
yazılıydı, düzeltildi:

```python
success = (not fell) and (not stuck)
fell  = gövde eğimi 60° üstünde 0,2 s sürerse (ya da ortam sonlandırırsa)
stuck = katedilen yol < 0.25 * mean|cmd_vx| * T      # cmd_vx > 0.05 ise
```

RMSE cinsinden bir eşik **yok**. Ve `cmd_vx ≈ 0` olduğunda mesafe koşulu
kapanıyor — `omni_lat`'ta yerinde duran politikanın 1,000 alması bundan
(1280 bölümde `stuck = 0,000` ile doğrulandı).

---

## 5. Doğrulanmış sonuçlar

Hepsi ham parquet/csv'den yeniden hesaplandı.

**Arazi ailesine göre başarı oranı** (19 senaryo, tohum 0–4):

| aile | PPO 1,475e9 | PPO 50M | FlashSAC 50M |
|---|---|---|---|
| zor engebe | 0,944 | 0,098 | 0,642 |
| eğim (6) | 0,665 | 0,836 | 0,711 |
| basamak (6) | 0,509 | 0,167 | 0,284 |
| **Ortalama** | **0,736** | **0,618** | **0,663** |

**Yanal hareket** (`omni_lat`, 0,5 m/s komut, 640 bölüm):

| kol | ulaşılan $v_y$ | RMSE $v_y$ | Ort.(19) |
|---|---|---|---|
| PPO 1,475e9 | 0,402 | 0,120 | 0,736 |
| PPO engebeli 50M | 0,001 | 0,500 | 0,618 |
| PPO düz 50M | 0,001 | 0,500 | — (tam matris koşulmadı) |
| FlashSAC engebeli 50M | 0,001 | 0,500 | 0,663 |
| **FlashSAC düz 50M** | **0,416** | **0,103** | 0,581 |

RMSE 0,500 = komuta eşit hata = hiç hareket yok (doygunluk değeri).

**Doygunluk:** FlashSAC `rough_hard` 42,5M'de 0,642 tepe; 200M koşusunda 0,559.
Gerileme tekdüze değil — en hızlı hücrede 0,322 → 0,069.

---

## 6. Şekiller

`TOK2026Conference/make_figs.py` — sadece diskteki parquet'i okur, hiçbir şeyi
yeniden hesaplamaz, yani tablolardan sapamaz.

```bash
cd TOK2026Conference && conda activate env_isaaclab && python make_figs.py
```

- `fig_rmse_isil.pdf` — Şekil 3, üç panelli RMSE ısıl haritası (`figure*`, tam sayfa)
- `fig3_cot.pdf` — Şekil 4, CoT (iki panel üst üste, tek sütun)
- `training_env_print.png` — Şekil 2, benzetim ekran görüntüsü (kontrastı açılmış)

**Tuzaklar (hepsi yaşandı):**
- Ondalık ayraç virgül olmalı. `FuncFormatter` yalnızca **tik** metnini yakalar;
  `ax.text()` hücreleri ve başlıklar `_tr()`'den geçmeli, `imshow` tikleri
  liste hâlinde verilmeli.
- Mathtext'te `$1{,}475$` yazılmalı — `$1,475$` yazarsanız TeX virgülü işleç
  sayıp boşluk koyar.
- matplotlib LaTeX ligatürü uygulamaz: `"--"` iki tire basar, yarım tire için
  gerçek `–` gerekir.
- `constrained_layout` varsayılan boşlukları bu en/boy oranında yetersiz;
  `fig.get_layout_engine().set(w_pad=..., wspace=...)` gerekiyor.

---

## 7. LaTeX tuzakları

- **`\usepackage[turkish,shorthands=off]{babel}` ZORUNLU.** Türkçe babel `=`
  karakterini aktif yapıyor, `\includegraphics[width=...]` bozuluyor. Önsözde
  `\shorthandoff{=}` yetmez.
- **Şekil genişliği `\tokcolw` (80 mm) ile verilmeli, `\columnwidth` ile değil**
  — bu şablonda `\columnwidth` 170 mm'ye çözülüyor.
- **`\includegraphics`'e `height` vermeyin**: 4:3 görselde yükseklik bağlayıcı
  olup `keepaspectratio` genişliği yarıya düşürüyor.
- TikZ'de `text width` bir TeX uzunluğu — birimsiz sayı **pt** sayılır,
  `text width=22.5` 22,5 mm değil 7,9 mm demektir.
- Sayfa genişliğinde (`figure*`) üçten fazla kayan nesne kuyruğa girip
  kaynakçadan sonraya düşüyor.

---

## 8. Sezgiye aykırı ama doğrulanmış olgular

Devralan kişi bunları tekrar keşfetmek zorunda kalmasın:

1. **PPO 4096 ortamla eğitildi, 256 ile değil.**
   `logs/rsl_rl/.../2026-08-07_12-48-55/params/env.yaml:81`. Dolayısıyla PPO'nun
   mini yığını 4096×24/4 = **24.576**, FlashSAC'inki 2048. Fark PPO'nun
   *lehine*; bildiri bunu Tablo 1'de olduğu gibi veriyor.

2. **FlashSAC'in öğrenilmiş sıcaklığı α = 5,8×10⁻⁴.**
   `flashsac_flat_s0_step195310/temperature.pt` içinden okundu. Yani entropi
   terimi eğitim sonunda pratikte **kapalı**. `std_bias` de 16 boyutta tekdüze
   (0,077–0,133). Bu yüzden "SAC entropiyle keşfi koruyor" mekanizması
   bildiriden **çıkarıldı** — kendi verimiz onu yalanlıyor.

3. **PPO'nun son denetim noktası gerçekten en iyisi.**
   `Train/mean_reward` 300. yinelemede 54,5 → 508'de ~75; arazi müfredatı
   0,478 → 1,817 (kesintisiz artıyor). `rough_hard@0,5`'te model_300'ün daha
   yüksek görünmesi bir yapaylık: 508 **hiç devrilmiyor** (0,000 vs 0,060) ama
   daha yavaş sürünüyor, o yüzden %25 mesafe eşiğini daha az geçiyor. İkisi de
   0,06–0,08 m/s gidiyor, yani ikisi de o arazide başarısız.

4. **Sabatini ve ark. [11] eşit bütçeli karşılaştırma YAPMIYOR.**
   Tam metin: *"the paper does not compare algorithms at equal environment-step
   budgets."* Bir ara bildiride bunun tersi yazılıydı, düzeltildi. Novelty
   iddiası hâlâ "tekerlekli-bacaklı"ya daraltılmış durumda — artık gerekçesi
   zayıf, genişletilebilir.

5. **PPO'nun tekrar oranı SAC'den yüksek.** SAC ≈ 4 çekim/geçiş
   (5e7 × 1,95e-3 × 2048 ≈ 2,0e8), PPO tam 5 (dönem sayısı). Yani örneklem
   verimliliğini "daha çok tekrar" ile açıklamak sayısal olarak eleniyor.

---

## 9. Değişmez kurallar

- **Ham deney verisi asla değiştirilmez.** Yeni değerlendirme yeni dizine yazar.
- **Sonuç/parquet git'e commit edilmez.**
- **Silmeden önce kullanıcı onayı.**
- tmux/nohup yok — ön planda ya da harness arka planında.
- Komutlar daima tam ve açık bayraklarla verilir.
- Ödül kümesi ve DR yapılandırması **değiştirilmez** — PPO üzerinde çalıştığı
  kanıtlanmış, adil karşılaştırma için sabit tutuluyor.

---

## 10. Açık işler

**Bildiride:**
- Novelty kapsamı kararı: "tekerlekli-bacaklı" (mevcut, güvenli) mi kalsın,
  "bacaklı"ya genişletilsin mi (madde 8.4 sayesinde artık dayanağı var).
- §1'de PPO için "güven bölgesi kısıtı" deniyor; PPO kırpma kullanır, güven
  bölgesi TRPO'nun. `"kırpma kısıtı"` daha güvenli.
- Kaynakça [17], [18], [20]–[23] (kullanıcının genişletme listesinden) hâlâ
  doğrulanmadı — yazar adı yok. **Doğrulamadan eklenmemeli**; bu bildiride daha
  önce iki sahte künye bulundu ve [19]'un başlığı da yanlış çıktı.

**Deneyde (isteğe bağlı):**
- PPO düz kolu tam matriste koşulmadı (yalnızca flat + omni, 3 senaryo).
  Tablo 3'teki `—` bundan. Bildirinin iddiası bunu gerektirmiyor.
- Tekrar belleği ablasyonu: bellekten yanal hareket içeren geçişleri çıkarıp
  yeteneğin kaybolup kaybolmadığına bakmak. Bildiriden çıkarılan mekanizma
  iddiasının sınanabilir hâli buydu.
