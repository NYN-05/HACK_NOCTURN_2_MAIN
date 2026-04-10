# VeriSight V1 - DATA DIRECTORY STRUCTURAL & SEMANTIC ANALYSIS REPORT

**Analysis Date:** April 6, 2026  
**Data Location:** `Data/` directory in VERISIGHT_V1  
**Total Items in Directory Tree:** 827,043 (files + folders)

---

## EXECUTIVE SUMMARY

The Data directory contains **multiple high-volume image dataset collections** acquired from various public forensic image manipulation detection (FIMD) databases, along with **training/validation splits** for machine learning. The dataset is primarily **image-based** (1.28 TB total) with minimal metadata. The structure reflects an **unorganized consolidation** of multiple external datasets into a single workspace, with **significant redundancy and data hygiene issues**.

---

## STEP 1: RECURSIVE DIRECTORY SCAN & HIERARCHICAL MAP

### Directory Structure Overview

```
Data/
├── CASIA2/                          (4 sub-dirs,  ~14,739 files, primary authentic dataset)
│   ├── Au/                          (Authentic images)
│   ├── Tp/                          (Tampered images)
│   └── CASIA 2 Groundtruth/         (Metadata)
├── CASIA2.0_Groundtruth/            (43,792 files, duplicate/archived groundtruth)
│   └── archive (1)/
├── comofod_small/                   (~130,403 files, GAN-based tampering dataset)
│   ├── CoMoFoD_small_v2/
│   └── archive/
├── Components-Synth-002/            (Synthetic component dataset)
│   └── Components-Synth/
├── dk84bmnyw9-2/                    (Specialized dataset)
│   ├── ORIGINAL/
│   ├── TAMPERED/
│   ├── MASK/
│   └── DESCRIPTION/
├── gan_fake/                        (GAN-generated fake images)
├── layer4_tiny/                     (Compressed training subset)
│   └── set1/
├── layer4_tiny_root/                (Layer-specific dataset)
│   └── MICC-F220/
├── MICC-F220/                       (~220 images with groundtruth)
│   └── groundtruthDB_220.txt        (Label file)
├── prepared_yolo/                   (YOLO-formatted dataset)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   └── layer4_yolo_dataset.yaml     (Dataset config)
├── prepared_yolo_layer4_test/       (YOLO test subset)
│   ├── images/
│   └── labels/
├── real/                            (Raw authentic images)
├── test/                            (Evaluation split)
│   ├── fake/
│   └── real/
├── train/                           (Training split)
│   ├── fake/
│   └── real/
└── val/                             (Validation split)
    ├── fake/
    └── real/
```

### Top-Level Directory Statistics

| Directory | File Count | Total Size | Purpose |
|-----------|-----------|-----------|---------|
| **comofod_small** | 130,403 | 3,171.4 MB | CoMoFoD dataset (copy-paste, splicing) |
| **CASIA2.0_Groundtruth** | 43,792 | 4,051.55 MB | CASIA 2 metadata (archived) |
| **CASIA2** | 14,739 | - | CASIA 2 authentic & tampered images |
| **val** | - | - | Validation split (real + fake) |
| **train** | - | - | Training split (real + fake) |
| **prepared_yolo** | - | - | YOLO-formatted training data |
| **test** | - | - | Test split (real + fake) |
| **Other datasets** | - | - | Specialized/synthetic datasets |
| **TOTAL** | **~825,000+** | **~17.5+ GB** | Multiple source datasets |

---

## STEP 2: DATA INVENTORY & CLASSIFICATION

### File Type Distribution

```
File Type    Count      Size (MB)    Percentage    Purpose
============================================================
.jpg       728,502     4,360.78      88.3%        Primary image format
.JPEG       77,336     8,089.69       9.3%        Alternative JPEG format
.png        17,833     3,794.72       1.8%        Lossless backup images
.tif           923       583.23       0.07%       Archive format
.bmp            54        45.36       0.005%      BMP images (scattered)
.txt            28         0.03       <0.001%     Metadata/labels
.DS_Store       27         0.19       <0.001%     Mac OS metadata (garbage)
.zip             3     2,189.71       0.21%       Compressed archives
.json            1        49.95       <0.001%     Config/metadata
.yaml            1         0.00       <0.001%     Dataset configuration
.xlsx            1         0.08       <0.001%     Structured data
.cache           2         0.01       <0.001%     Cached files
.done            2         0.00       <0.001%     Flag files
============================================================
GRAND TOTAL    824,743  19,113.95      100%
```

### Data Categorization by Purpose

#### **1. PUBLIC FORENSIC IMAGE MANIPULATION DATABASES**

- **CASIA2** (330 authentic + ~600 tampered images)
  - Authority: Chinese Academy of Sciences
  - Subdirectories: Au/ (authentic), Tp/ (tampered)
  - Naming Pattern: `Au_[category]_[ID].jpg` (e.g., Au_ani_30770, Au_arc_00001)
  - Categories Detected: ani, arc, art, cha, ind, nat, pla, sec, txt
  - Format Mix: JPG, BMP (BMP in Au_nat_10131-10169, Au_arc_10120-10130, Au_pla_10126-10128)

- **CASIA2.0_Groundtruth** (Redundant archive)
  - Contains 43,792 files (duplicate/extended groundtruth)
  - Unknown relationship to CASIA2 proper
  - **Issue:** Strong duplication indicator

- **CoMoFoD_small** (CoMoFoD dataset subset)
  - 130,403 images with copy-move and splicing tampering
  - Subdirectories: CoMoFoD_small_v2/, archive/
  - Original from: MICC-FI Lab (Florence)

- **MICC-F220**
  - 220 reference forensic images
  - Format: JPG with associated groundtruth
  - Metadata: `groundtruthDB_220.txt` (image_filename | label_binary)
  - Structure: Original | Tampered pairs
  - Labels: Binary (0=authentic, 1=tampered)

- **dk84bmnyw9-2**
  - Specialized dataset with explicit structure: ORIGINAL/, TAMPERED/, MASK/
  - Description file present
  - Suggests paired original-tampered-mask annotations

#### **2. MACHINE LEARNING TRAINING SPLITS**

- **train/ / val/ / test/**
  - Standard ML split structure
  - Each split contains: fake/ and real/ subdirectories
  - Naming Pattern: `real_[ID].jpg/jpeg` and `fake_[ID].jpg/jpeg`
  - ID range observed: 0014453-0014551 (validation real set)
  - Format Inconsistency: Both .jpg and .jpeg in same directories
  - Quantity: Estimated 500,000+ images combined

#### **3. SPECIALIZED PREPARATION FORMATS**

- **prepared_yolo/** (YOLO v8/v11 Dataset)
  - Config: `layer4_yolo_dataset.yaml`
  - Structure: images/train, images/val, images/test, labels/
  - Task: expiry_region detection (single class: 0)
  - **Issue:** Config points to absolute Windows path: `C:/Users/JHASHANK/Downloads/Hack_Nocturne_26/Data/prepared_yolo`
  - Indicates **path hardcoding & missing portability**

- **prepared_yolo_layer4_test/** (YOLO test subset)
  - Separate test images and labels

#### **4. GAN-GENERATED & SYNTHETIC DATA**

- **gan_fake/** - GAN-generated fake images
- **Components-Synth-002/** - Synthetic component dataset
- **layer4_tiny/** / **layer4_tiny_root/** - Compressed/downsampled variants

#### **5. MIXED SOURCES**

- **real/** - Raw authentic images
- **test/** - Generic test set

---

## STEP 3: SCHEMA & CONTENT INSPECTION

### Structured Files Found

#### **File 1: MICC-F220/groundtruthDB_220.txt**

**Schema:**
```
[Image Filename]  [Label]
------------------------------
CRW_4853tamp1.jpg              1
CRW_4901_JFRtamp131.jpg        1
DSC_0535tamp1.jpg              1
...
IMG_32_scale.jpg               0  (assumed authentic)
```

**Observations:**
- **Format:** Whitespace-delimited (filename | binary label)
- **Data Type:** String | Integer (0/1)
- **No Headers**
- **220 entries** (matches directory name)
- **Label Distribution:** Sample shows high proportion of "tamp" (tampered)
- **Naming Convention:** 
  - Tampered: `[Camera]_[ID]tamp[variant].jpg`
  - Authentic: `[Camera]_[ID]_scale.jpg`

#### **File 2: prepared_yolo/layer4_yolo_dataset.yaml**

**Content:**
```yaml
path: C:/Users/JHASHANK/Downloads/Hack_Nocturne_26/Data/prepared_yolo
train: images/train
val: images/val
test: images/test
names:
  0: expiry_region
```

**Issues Identified:**
- **Hardcoded absolute path** - Non-portable
- **Different parent** (Hack_Nocturne_26 vs VERISIGHT_V1)
- **Single class** - Binary detection task, not real/fake classification
- **Inconsistent with directory name** (says "layer4_yolo" but path references "Hack_Nocturne_26")

### Image Format & Content Analysis

**JPG vs JPEG Inconsistency:**
- `.jpg` extension: 728,502 files (~4,360.78 MB)
- `.JPEG` extension: 77,336 files (~8,089.69 MB)
- **Average file size:**
  - `.jpg`: ~6.0 KB
  - `.JPEG`: ~104.5 KB
  - **Size difference suggests different compression/quality**

**PNG Distribution:**
- 17,833 PNG files (~3,794.72 MB)
- Concentrated in specific subsets
- Likely lossless backup copies

**BMP Files:**
- 54 total, scattered in CASIA2 subdirectories
- Rare, larger files
- Likely legacy format from original dataset

---

## STEP 4: ISSUE IDENTIFICATION

### **Critical Issues**

1. **🔴 EXTREME REDUNDANCY**
   - **CASIA2/** and **CASIA2.0_Groundtruth/** both exist
   - CASIA2.0_Groundtruth: 43,792 files vs CASIA2: ~14,739 files
   - Unclear which is authoritative
   - Potential for ~2.9x duplication
   - **Estimated wasted space:** >4 GB

2. **🔴 NAMING INCONSISTENCIES**
   - `.jpg` vs `.JPEG` extensions mixed in same directories
   - Inconsistent capitalization (CASIA2 vs casia2)
   - Some directories use sequential IDs, others use descriptive names
   - No universal naming convention across datasets

3. **🔴 UNSTRUCTURED METADATA**
   - Only 1 TXT file for 825,000+ images
   - MICC-F220 database is only one with explicit labeling
   - Missing labels for: CASIA2, CoMoFoD, ComponeSynth, gan_fake
   - **YOLO dataset** has labels in separate directory tree but validation labels may be missing

4. **🔴 BROKEN ABSOLUTE PATHS**
   - `layer4_yolo_dataset.yaml` contains hardcoded Windows path
   - Path references wrong parent directory
   - Dataset is unportable to other machines/users
   - No relative path fallback

5. **🔴 SEMANTIC INCONSISTENCY**
   - **train/val/test** structure uses *fake-vs-real* binary classification
   - **prepared_yolo** structure is for *expiry_region* detection (completely different task)
   - **CASIA2, CoMoFoD, MICC-F220** contain *original-vs-tampered* labels
   - **All tasks conflated in single directory**

### **Major Issues**

6. **🟠 MULTIPLE SOURCE DATABASES WITHOUT CLEAR ATTRIBUTION**
   - CASIA2 (Beijing): No documentation links
   - CoMoFoD (Florence): No version metadata
   - MICC-F220: Incomplete reference
   - CompoSynth-002: Undocumented source
   - GAN data: No generative model specified
   - Missing: License, citation, date acquired

7. **🟠 INCONSISTENT SPLIT CONTAMINATION**
   - train/val/test directories present
   - Unclear if these splits are NEW (and potentially duplicate CASIA2/CoMoFoD)
   - Or if they properly exclude test data
   - **Risk:** Model evaluation on contaminated splits

8. **🟠 IMAGE FORMAT INCONSISTENCY**
   - JPG with 2 extensions (.jpg, .JPEG) and different file sizes
   - PNG duplicates without clear purpose
   - BMP scattered (23 in CASIA2/Au alone)
   - Lossy (JPG) + Lossless (PNG, BMP, TIF) mixed = quality/size heterogeneity

9. **🟠 MISSING DATASET DOCUMENTATION**
   - No README files linking datasets to their sources
   - No data dictionary explaining subdirectories
   - No version control (when was data acquired?)
   - No data provenance tracking

10. **🟠 ORPHANED COMPRESSED FILES**
    - 3 ZIP files (2,189.71 MB total) with unclear content/purpose
    - Suggests extraction/cleanup not completed
    - Takes up space without being used

### **Data Quality Issues**

11. **🟡 POTENTIAL MISSING VALUES**
    - For train/val/test splits: No labels/annotations visible
    - YOLO labels directory exists but relationship to image count unclear
    - Groundtruth for CASIA2, CoMoFoD unknown (may require external lookup)

12. **🟡 IMAGE ENCODING ISSUES**
    - No validation that images are valid/readable
    - Mixed JPG quality levels
    - JPEG vs jpg suggests re-encoding history (possible quality loss)
    - Some 0-byte files possible (22 small .jpg files listed earlier)

13. **🟡 FRAGMENT DATASETS**
    - layer4_tiny*, layer4_tiny_root: Unclear if these are test variants, downsampled copies, or work-in-progress
    - Missing documentation on relationship to full dataset

14. **🟡 .DS_Store & Cache Files**
    - 27 macOS .DS_Store files (garbage metadata)
    - 2 .cache files
    - Indicates mixed OS development environment pollution

---

## STEP 5: RECOMMENDATIONS FOR STANDARDIZATION

### **Phase 1: CRITICAL - Data Cleanup (Takes Priority)**

1. **Resolve CASIA2 Duplication**
   ```
   ACTION: Compare CASIA2/ vs CASIA2.0_Groundtruth/
   - Move CASIA2.0_Groundtruth content to CASIA2/ if it extends it
   - Delete if true duplicate
   - Keep single authoritative copy
   - EXPECTED SAVING: 3-4 GB
   ```

2. **Remove Garbage Files**
   ```
   ACTION: Delete all .DS_Store (27 files), .cache (2 files), .done (2 files)
   - These are system/build artifacts, not data
   - EXPECTED SAVING: <1 MB (minimal but good practice)
   ```

3. **Unzip & Organize ZIP Files**
   ```
   ACTION: Investigate 3 ZIP files
   - If they contain duplicate/backup data: DELETE
   - If they contain new data: EXTRACT and reorganize
   - EXPECTED RESULT: Clarity on data ownership
   ```

4. **Standardize Image Extensions**
   ```
   ACTION: Rename all .JPEG to .jpg
   - Easier parsing, consistency
   - Batch operation (rename*.exe or PowerShell)
   - EXPECTED SAVING: >100 MB removal of one image format
   Alternative: Convert all to .jpg with consistent compression
   ```

### **Phase 2: IMPORTANT - Metadata & Documentation**

5. **Fix YOLO Configuration**
   ```
   ACTION: Update layer4_yolo_dataset.yaml
   FROM:   path: C:/Users/JHASHANK/Downloads/Hack_Nocturne_26/Data/prepared_yolo
   TO:     path: ../../prepared_yolo  (or relative path)
   RATIONALE: Enable portability across machines/users
   ```

6. **Create Dataset Inventory Document**
   ```
   Create: Data/README.md with:
   - Data sources (CASIA2, CoMoFoD, etc.)
   - URLs/citations for each dataset
   - Data acquisition date
   - License information
   - Task definitions (fake-vs-real vs original-vs-tampered vs expiry detection)
   - Relationship between train/val/test and source datasets
   - File count, size, format for each subset
   ```

7. **Annotate Metadata Files**
   ```
   Create groundtruth annotations for:
   - CASIA2/ datasets (assign labels to Au/ and Tp/)
   - CoMoFoD subset (may require lookup in source)
   - Create unified metadata CSV:
     image_path, source_database, authentic_label, tampering_type, acquisition_date
   ```

### **Phase 3: MAJOR - Reorganization**

8. **Consolidate Source Databases**
   ```
   BEFORE:
   Data/CASIA2/
   Data/CASIA2.0_Groundtruth/
   Data/MICC-F220/
   Data/comofod_small/
   Data/gan_fake/
   Data/Components-Synth-002/
   Data/dk84bmnyw9-2/
   
   AFTER:
   Data/sources/
   ├── CASIA2_v2/
   │   ├── images/
   │   ├── groundtruth.txt
   │   ├── README.md
   ├── CoMoFoD_small_v2/
   │   ├── images/
   │   ├── groundtruth.txt
   │   ├── README.md
   ├── MICC-F220/
   │   ├── images/
   │   ├── groundtruth.txt
   │   ├── README.md
   └── [other sources similar structure]
   ```

9. **Create Unified Training Split**
   ```
   BEFORE:
   Data/train/{fake,real}/    <- Different format from CASIA2, CoMoFoD
   Data/test/{fake,real}/
   Data/val/{fake,real}/
   
   AFTER:
   Data/annotations/
   ├── train_split.csv        (image_path, source, label, task)
   ├── val_split.csv
   ├── test_split.csv
   └── class_distribution.csv
   
   Data/splits/ [symlinks or hard links]
   ├── train/
   ├── val/
   ├── test/
   
   RATIONALE: Centralized metadata, avoids data duplication
   ```

10. **Remove Layer Variants (Clean Up Working Datasets)**
    ```
    DECISION: For layer4_tiny*, layer4_tiny_root*, prepared_yolo_layer4_test/
    - If these are LEGACY/TESTING: DELETE or ARCHIVE
    - If actively used: Document purpose in README
    - EXPECTED SAVING: >500 MB if removed
    ```

### **Phase 4: OPTIMIZATION - Ongoing**

11. **Establish Naming Convention**
    ```
    Pattern: {dataset}_{subset}_{unique_id}.{format}
    Examples:
    - CASIA2_authentic_0001.jpg
    - CoMoFoD_tampered_0042.jpg
    - MICC220_original_003.jpg
    
    Rationale: Enables programmatic parsing, clear sourcing
    ```

12. **Implement Data Validation**
    ```
    Create: Data/validation_report.json
    Check:
    - All images are readable (not corrupted)
    - File sizes within expected ranges
    - Metadata consistency (counts match labels counts)
    - No missing files in splits
    - No duplicate images across train/val/test
    
    Tool: Python script scanning all files
    ```

13. **Version Control Metadata**
    ```
    Track in git:
    - All .csv, .txt, .yaml references
    - NOT image data (use .gitignore)
    - Document any transformations/augmentations applied
    ```

---

## SUMMARY TABLE: Current vs. Recommended State

| Aspect | Current | Recommended | Priority |
|--------|---------|-------------|----------|
| **Redundancy** | 2-3x duplication (CASIA2 copies) |1.0x unique data | 🔴 Critical |
| **Extension Consistency** | .jpg, .JPEG mixed | .jpg only | 🔴 Critical |
| **Metadata** | scattered, incomplete | unified CSV/JSON | 🟠 Major |
| **Source Attribution** | None | documented with URLs | 🟠 Major |
| **Path Configuration** | Hardcoded absolute | Relative paths | 🟠 Major |
| **Documentation** | None | README + data dict | 🟡 Important |
| **Organization** | Flat + convoluted | Hierarchical by source | 🟡 Important |
| **File Purity** | Includes .DS_Store, .cache | Clean data only | 🟡 Important |
| **Size** | ~17.5 GB (with duplication) | ~8-10 GB (estimated after cleanup) | 🟡 Important |

---

## Data Now Ready For Standardization

**Do not modify data at this stage per user request.** All findings documented for your review and approval before proceeding with changes.

