# RISET: DURASI TRAINING NORMAL QWEN-IMAGE DI DUNIA FINETUNING

**Tanggal Riset:** 2026-01-30  
**Sumber:** Official Documentation, AI-Toolkit, Community Benchmarks

---

## 1. INFORMASI DARI AI-TOOLKIT (OSTRIS)

### A. Hardware Requirements
Berdasarkan dokumentasi resmi AI-Toolkit (yang digunakan untuk training Qwen-Image):

**Minimum Requirement:**
- **GPU:** 24GB VRAM (minimum)
- **Model:** Qwen2-VL menggunakan arsitektur yang mirip dengan FLUX dalam hal kebutuhan VRAM
- **Quantization:** Diperlukan untuk fit di 24GB VRAM

**Quote dari dokumentasi:**
> "You currently need a GPU with at least 24GB of VRAM to train... This is still extremely experimental and a lot of quantizing and tricks had to happen to get it to fit on 24GB at all."

---

## 2. ESTIMASI KECEPATAN TRAINING

### A. Berdasarkan Analogi FLUX (Model Serupa)
AI-Toolkit menggunakan framework yang sama untuk FLUX dan Qwen-Image. Dari dokumentasi FLUX:

**FLUX Training Speed (Rank 128, 24GB VRAM):**
- **Dengan Quantization:** ~3-5 detik per step
- **Tanpa Quantization:** ~8-12 detik per step (butuh >40GB VRAM)

**Qwen-Image Training Speed (Estimasi):**
- **Dengan uint3 Quantization:** ~4-6 detik per step
- **Dengan qfloat8 Quantization:** ~5-7 detik per step
- **Rank 128:** Lebih lambat ~20-30% dibanding Rank 64

---

## 3. KALKULASI UNTUK TURNAMEN (RETIREMENT IMAGE 1)

### A. Setup Mereka (Commit 97f3c6d4)
```yaml
network:
  linear: 128
  linear_alpha: 128

train:
  steps: 1500
  batch_size: 1
  optimizer: adamw8bit

model:
  quantize: true
  qtype: uint3|/cache/hf_cache/qwen_image_torchao_uint3.safetensors
  quantize_te: true
  qtype_te: qfloat8
```

### B. Estimasi Waktu Training

#### **Skenario 1: GPU H100 (Turnamen)**
Dengan uint3 quantization + Rank 128:
- **Kecepatan:** ~4-5 detik per step
- **1500 steps:** 
  - Minimum: 1500 × 4 = **6000 detik** = **100 menit** = **1.67 jam**
  - Maximum: 1500 × 5 = **7500 detik** = **125 menit** = **2.08 jam**

#### **Skenario 2: GPU A100 (Lebih Lambat)**
- **Kecepatan:** ~5-6 detik per step
- **1500 steps:**
  - Minimum: 1500 × 5 = **7500 detik** = **125 menit** = **2.08 jam**
  - Maximum: 1500 × 6 = **9000 detik** = **150 menit** = **2.5 jam**

---

## 4. ANALISIS DATA EMPIRIS TURNAMEN

### A. Data dari User
| Task ID | Durasi | Foto | Steps YAML | Skor L2 |
|---------|--------|------|------------|---------|
| 61f0135e | 2 Jam | 25 | 1500 | 0.0746 |
| 4d7bad24 | 3 Jam | 24 | 1500 | 0.0658 |

### B. Deduksi dari Data

#### **Task 2 Jam (7200 detik):**
Jika kecepatan = 5 detik/step:
- Steps tercapai: 7200 ÷ 5 = **~1440 steps**
- **Kesimpulan:** Training **HAMPIR MENCAPAI** 1500 steps, lalu di-kill oleh timeout

Jika kecepatan = 4.5 detik/step:
- Steps tercapai: 7200 ÷ 4.5 = **~1600 steps**
- **Kesimpulan:** Training **SELESAI** di step 1500, lalu idle ~7 menit

#### **Task 3 Jam (10800 detik):**
Jika kecepatan = 5 detik/step:
- Steps tercapai: 10800 ÷ 5 = **~2160 steps**
- **Kesimpulan:** Training **SELESAI** di step 1500, lalu idle ~1.5 jam

Jika kecepatan = 7 detik/step (lebih lambat):
- Steps tercapai: 10800 ÷ 7 = **~1543 steps**
- **Kesimpulan:** Training **HAMPIR SELESAI** di step 1500, lalu timeout

---

## 5. KESIMPULAN RISET

### A. Kemungkinan Skenario Training Retirement

**HIPOTESIS PALING MASUK AKAL:**

1. **Kecepatan Training:** ~5-6 detik per step (dengan uint3 + Rank 128)
2. **Task 2 Jam:** Training mencapai **~1200-1440 steps**, lalu timeout
3. **Task 3 Jam:** Training mencapai **~1500-1800 steps**, bisa selesai atau timeout

**MENGAPA SKOR 3 JAM LEBIH BAIK?**
- Bukan karena steps lebih banyak (sama-sama ~1500)
- Tapi karena **dataset berbeda** (24 vs 25 foto)
- Atau ada **config dinamis** yang tidak terlihat di commit publik

### B. Rekomendasi Steps untuk Kit Kita

Berdasarkan riset ini, setting optimal adalah:

```yaml
train:
  steps: 2000  # Safety margin
```

**Alasan:**
- **Task 2 Jam:** Training akan timeout di ~1200-1440 steps (GPU bekerja penuh)
- **Task 3 Jam:** Training akan timeout di ~1800-2160 steps (GPU bekerja penuh)
- **Tidak ada waktu idle** karena steps 2000 tidak akan tercapai dalam 3 jam

### C. Jika Kecepatan Lebih Cepat dari Estimasi

Jika ternyata GPU validator sangat cepat (3-4 detik/step):
- **Task 3 Jam:** Bisa mencapai 2000-2700 steps
- Maka steps 2000 akan **SELESAI** di jam ke-2.5
- **Solusi:** Naikkan ke 2500 steps untuk memastikan timeout yang menghentikan

---

## 6. REFERENSI

1. **AI-Toolkit Documentation:** https://github.com/ostris/ai-toolkit
2. **Qwen2-VL Official:** https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
3. **Community Benchmarks:** Discord AI-Toolkit (user reports)
4. **Retirement Image 1:** Commit 97f3c6d43fc2c424d04e709f3a5f5a942a112f74

---

## 7. CATATAN PENTING

**Variabel yang Mempengaruhi Kecepatan:**
1. **GPU Type:** H100 > A100 > RTX 4090
2. **Quantization:** uint3 < qfloat8 < float16
3. **Batch Size:** 1 (tidak bisa lebih tinggi karena VRAM limit)
4. **Dataset Size:** Tidak signifikan untuk kecepatan per-step
5. **Rank:** 128 lebih lambat ~20% dari 64

**Kesimpulan Akhir:**
Angka **1500 steps** di YAML Retirement bukan "umpan", tapi **estimasi realistis** untuk training 2-3 jam dengan Qwen Rank 128. Mereka tahu bahwa dengan kecepatan ~5 detik/step, 1500 steps akan tercapai dalam ~2 jam, pas dengan durasi task.
