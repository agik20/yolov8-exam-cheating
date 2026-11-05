# Implementation Plan — YOLOv8 Exam Cheating Detection System

**Last Updated:** 2026-08-19
**Status:** DRAFT — Waiting for execution approval
**Goal:** Refactor dan optimasi pipeline end-to-end dari data collection hingga TensorRT inference

---

## Overview

Project ini adalah sistem deteksi kecurangan ujian berbasis YOLOv8 (pose + object detection) dengan deployment TensorRT. Analisis komprehensif mengungkap **5 kategori masalah kritis** yang menghambat reproduktibilitas, stabilitas, dan performa:

1. **Hardcoded paths** di 15+ file — pipeline tidak bisa berjalan di mesin lain
2. **Konsistensi kelas** — 3 definisi class yang bertentangan di inference vs training config
3. **Double NMS** — over-filtering yang mengurangi recall
4. **Tidak ada dependency management** — tidak ada `requirements.txt` atau `.gitignore`
5. **Kurang error handling & monitoring** — pipeline rentan crash tanpa fallback

Rencana ini membagi perbaikan menjadi **4 Phase** dengan urutan eksekusi yang bergantung satu sama lain.

---

## Phase 1: Foundation & Structure (Pondasi)

**Tujuan:** Menyiapkan struktur project yang reproducible dan portable.

### Step 1.0 — Buat Project Folder Baru
- **Action:** Buat folder project baru di `C:\Users\ardut\projects\yolov8-exam-cheat-detection`
- **PENTING:** Folder lama `C:\Users\ardut\projects\yolov8-exam-cheating` TIDAK DISentuh — tetap ada sebagai reference/source code
- **Strategi:** Copy seluruh isi folder lama ke folder baru, lalu refactor code di folder baru tersebut
- **Command:**
  ```bash
  mkdir C:/Users/ardut/projects/yolov8-exam-cheat-detection
  cp -r C:/Users/ardut/projects/yolov8-exam-cheating/* C:/Users/ardut/projects/yolov8-exam-cheat-detection/
  cd C:/Users/ardut/projects/yolov8-exam-cheat-detection
  ```
- **Post-copy action:** Hapus `yolov8-exam-cheating.zip` di folder baru (sudah tidak diperlukan), hapus `.git` agar repo baru bisa diinisialisasi ulang
- **Risk:** Low — folder lama aman, semua perubahan hanya di folder baru
- **Time estimate:** 5 menit

### Step 1.1 — Buat `requirements.txt`
- **File:** `requirements.txt` (new)
- **Dependencies:** `ultralytics>=8.0`, `opencv-python>=4.8`, `flask>=3.0`, `albumentations>=1.3`, `numpy>=1.24`, `pandas>=2.0`, `scikit-learn>=1.3`, `torch>=2.0`, `onnx>=1.14`, `tqdm>=4.65`
- **Risk:** Low — hanya deklarasi, tidak mengubah logic
- **Time estimate:** 15 menit

### Step 1.2 — Buat `.gitignore`
- **File:** `.gitignore` (new)
- **Entries:** `__pycache__/`, `*.pyc`, `*.pyo`, `runs/`, `inference/output/`, `*.engine`, `*.pt`, `*.onnx`, `*.csv`, `*.txt` (kecuali config), `.env`, `venv/`, `node_modules/`
- **Risk:** Low
- **Time estimate:** 10 menit

### Step 1.3 — Buat `config/settings.yaml`
- **File:** `config/settings.yaml` (new)
- **Konten:** Centralized configuration — semua path (model, data, output), hyperparameter (conf_thresh, iou_threshold, cooldown), dan metadata project
- **Benefit:** Satu sumber kebenaran untuk semua konfigurasi
- **Risk:** Low — additive, tidak menghapus apa pun
- **Time estimate:** 30 menit

### Step 1.4 — Refactor path resolution
- **Files:** Semua `.py` dan `.yaml` yang mengandung hardcoded path
- **Strategi:** Ganti `C:\Users\HeimTech\...` dan `/media/heimy/...` dengan path relatif berbasis `Path(__file__).parent` + settings.yaml
- **Scope:** ~15 file perlu diupdate
- **Risk:** Medium — harus tested per-file, salah path = script error
- **Time estimate:** 2-3 jam

### Step 1.5 — Buat `README.md`
- **File:** `README.md` (new)
- **Konten:** Architecture overview, setup instructions, directory structure explanation, how to run each pipeline stage
- **Risk:** Low
- **Time estimate:** 1 jam

---

## Phase 2: Data Pipeline Cleanup

**Tujuan:** Membersihkan dan menstandarisasi data collection, preprocessing, dan annotation scripts.

### Step 2.1 — Fix `augmentation-classwise.py`
- **Line 126:** Ganti `if transform.transforms[0].p > 0.5:` dengan `if random.random() < 0.5:` untuk konsistensi dengan `A.HorizontalFlip(p=0.5)`
- **Risk:** Low — bug fix
- **Time estimate:** 15 menit

### Step 2.2 — Fix `only-one-subject-auto-annotations.py`
- **Line 45:** Hapus `or None or` yang merupakan syntax error copy-paste
- **Risk:** Low — bug fix
- **Time estimate:** 5 menit

### Step 2.3 — Standardize label visibility logic
- **Masalah:** Beberapa script menggunakan threshold berbeda untuk `v` (visibility): `0.05`, `0.5`, hardcoded
- **Solusi:** Buat helper function `normalize_visibility(conf, low=0.05, high=0.5)` di module utilitas
- **Files affected:** `trained-model-auto-annotations.py`, `only-one-subject-auto-annotations.py`
- **Risk:** Medium — mengubah format label, perlu validasi ulang
- **Time estimate:** 1 jam

### Step 2.4 — Buat `utils/` module
- **Files:** `utils/__init__.py`, `utils/path_utils.py`, `utils/label_utils.py`, `utils/nms_utils.py`
- **Konten:** Path resolution helpers, label parsing/formatting, shared NMS implementation
- **Risk:** Low — baru dibuat, tidak mengubah existing code
- **Time estimate:** 1.5 jam

---

## Phase 3: Training & Evaluation Consistency

**Tujuan:** Menstandarisasi konfigurasi training dan evaluasi.

### Step 3.1 — Fix YAML configs
- **Files:** `training-datasets/exam-act/config.yaml`, `cheating-keypoint-recognition.yaml`, `smartphone-detection.yaml`
- **Perbaikan:**
  - Samakan `nc` dengan jumlah `names` (3 untuk pose, 1 untuk object)
  - Tambahkan `test` split
  - Pastikan `kpt_shape: [17, 3]` konsisten
  - Gunakan path relatif
- **Risk:** Medium — perlu validasi bahwa dataset sesuai config baru
- **Time estimate:** 30 menit

### Step 3.2 — Tambahkan test split configuration
- **Files:** Buat `test.yaml` untuk pose dan object detection
- **Konten:** Train/val/test paths, class definitions
- **Risk:** Low
- **Time estimate:** 20 menit

### Step 3.3 — Standardisasi evaluation scripts
- **Files:** `evaluasi-model/mAP-measurements.py`, `evaluate_oks.py`, `classification-eval.py`
- **Perbaikan:** Ganti semua hardcoded path dengan path argument atau settings.yaml
- **Risk:** Low
- **Time estimate:** 30 menit

### Step 3.4 — Buat training experiment tracker
- **File:** `training-datasets/exam-act/train.py` (unified trainer)
- **Fitur:** W&B logging, best-model checkpointing (based on mAP50), resume from checkpoint
- **Risk:** Medium — new feature, perlu testing
- **Time estimate:** 3 jam

---

## Phase 4: Inference Pipeline Fixes

**Tujuan:** Memperbaiki bug critical di inference dan meningkatkan robustness.

### Step 4.1 — Fix class mapping mismatch (CRITICAL)
- **File:** `inference/main.py`
- **Masalah:** `Config.POSE_CLASSES`, `pose_class_map`, dan YAML `names` tidak konsisten
  - `Config.POSE_CLASSES = ["normal", "sleep", "eat and drink"]`
  - `pose_class_map = {0: 'no-cheating', 1: 'provide object', 2: 'see friends work'}`
  - YAML names: `no-cheating`, `provide-object`, `see-friends-work`
- **Solusi:** Satukan ke satu source of truth — definisikan class names di `config/settings.yaml`, import di semua module
- **Risk:** High — mengubah behavior inference, HARUS testing menyeluruh
- **Time estimate:** 1 jam

### Step 4.2 — Remove double NMS
- **File:** `inference/main.py`
- **Masalah:** `model.track(iou=0.45)` + `apply_nms()` menghasilkan over-filtering
- **Solusi:** Pilih salah satu — hapus custom `apply_nms()` dan gunakan YOLO built-in NMS saja (lebih optimal karena sudah di-trt)
- **Risk:** Medium — perubahan post-processing logic
- **Time estimate:** 30 menit

### Step 4.3 — Tambahkan error handling
- **File:** `inference/main.py`
- **Perbaikan:**
  - Wrap pose/object detection dalam try/except
  - Fallback: skip frame jika model error, tetap process frame berikutnya
  - Log error dengan timestamp ke file log
- **Risk:** Low — defensive coding
- **Time estimate:** 30 menit

### Step 4.4 — Parameterize video & model paths
- **File:** `inference/main.py`
- **Perbaikan:** Baca `VIDEO_PATH`, `POSE_MODEL_PATH`, `OBJECT_MODEL_PATH` dari `config/settings.yaml`
- **Risk:** Medium — path harus valid saat runtime
- **Time estimate:** 30 menit

### Step 4.5 — Tambahkan model validation on startup
- **File:** `inference/main.py`
- **Perbaikan:** Sebelum start pipeline, cek keberadaan file model dan video. Tampilkan error message jelas jika file tidak ditemukan.
- **Risk:** Low
- **Time estimate:** 15 menit

### Step 4.6 — Thread-safe frame queue
- **File:** `inference/main.py`
- **Perbaikan:** Pastikan `frame_queue` operations thread-safe, tambahkan timeout yang合理
- **Risk:** Low
- **Time estimate:** 20 menit

---

## Impact & Risk Assessment

| Phase | Impact | Risk | Effort | Priority |
|-------|--------|------|--------|----------|
| **1: Foundation** | 🔴 Critical — tanpa ini tidak ada yang bisa jalan di mesin lain | 🟡 Medium — path refactor berisiko salah | 4-5 jam | **P1** |
| **2: Data Pipeline** | 🟡 High — memperbaiki augmentasi & annotation quality | 🟢 Low — bug fixes terisolasi | 2-3 jam | **P2** |
| **3: Training** | 🟡 High — konsistensi config & experiment tracking | 🟡 Medium — perlu validasi dataset | 4-5 jam | **P3** |
| **4: Inference** | 🔴 Critical — fix class mapping & double NMS langsung affect accuracy | 🔴 High — perubahan logic detection | 2-3 jam | **P1** |

### Dependency Map (urutan wajib):
```
Phase 1 (Step 1.3: settings.yaml)
    ↓
Phase 1 (Step 1.4: path refactor) + Phase 4 (Step 4.4: parameterize paths)
    ↓
Phase 2 (Steps 2.1-2.4) + Phase 3 (Steps 3.1-3.3)
    ↓
Phase 4 (Steps 4.1-4.6: inference fixes)
```

### Key Risk Mitigations:
1. **Path refactor** — Test setiap script dengan `python -c "import script"` sebelum ganti path
2. **Class mapping** — Buat test fixture dengan 3 sample image + label untuk validasi
3. **Double NMS removal** — Bandingkan hasil deteksi sebelum-sesudah di 10 sample frame
4. **All changes** — Gunakan branch isolasi, jangan langsung edit branch utama

---

## Execution Recommendation

Mulai dari **Phase 1, Step 1.1** (requirements.txt) karena ini paling aman dan foundational. Setelah itu lanjut ke **Step 1.3** (settings.yaml) sebagai single source of truth, baru lanjut path refactoring.

**Total estimated time:** 12-16 jam kerja fokus.

---

**Status:** Plan ready. Menunggu approval untuk eksekusi step mana yang mau dimulai.
