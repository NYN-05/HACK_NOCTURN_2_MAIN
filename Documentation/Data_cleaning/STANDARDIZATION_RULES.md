# DATA STANDARDIZATION RULES

**Document Version:** 1.0  
**Effective Date:** 2026-04-06  
**Scope:** VeriSight V1 Data Directory Cleaning

---

## 1. FILE NAMING CONVENTION

### General Rules
- **Format:** `{dataset}_{subset}_{unique_id}.{extension}`
- **Case:** All lowercase
- **Separators:** Underscores for word separation, no spaces
- **Invalid Characters:** No special chars except underscore
- **Examples:**
  - ✅ `casia2_authentic_0001.jpg`
  - ✅ `comofod_splicing_0042.jpg`
  - ❌ `CASIA2 - Authentic 0001.jpg`
  - ❌ `CoMoFoD@Splicing#0042.jpg`

### Image File Extensions
- **Standard Format:** `.jpg` (lowercase)
- **Action:** Convert all `.JPEG` → `.jpg`
- **Backup Formats:** `.png`, `.tif` (keep if intentional, document reason)
- **Policy:** One extension per format, no duplicates

### Metadata Files
- **Format:** `{dataset}_metadata.csv` or `.json`
- **Examples:**
  - `casia2_metadata.csv`
  - `comofod_groundtruth.csv`
  - `micc220_labels.csv`

---

## 2. COLUMN NAMING (CSV/JSON)

### Style
- **Format:** `snake_case`
- **Lowercase:** All letters lowercase
- **No Spaces:** Use underscores
- **Examples:**
  - ✅ `image_path`, `source_dataset`, `label_authentic`, `tampering_type`
  - ❌ `ImagePath`, `Source Dataset`, `Label (Authentic)`

### Standard Column Set (Universal Metadata)
```
image_path              : str    (relative path from dataset root)
source_dataset          : str    (CASIA2, CoMoFoD, MICC-F220, etc.)
filename_original       : str    (original filename)
authentic               : int    (1=authentic, 0=tampered/manipulated)
tampering_type          : str    (copy_move, splicing, gan, synthesis, none)
quality_level           : str    (high, medium, low)
resolution              : str    (width x height, e.g., "512x512")
file_format             : str    (jpg, png, tif)
file_size_bytes         : int    (uncompressed size)
acquisition_date        : str    (YYYY-MM-DD or null)
processing_notes        : str    (any transformations applied)
```

---

## 3. DATE FORMAT

### Standard: ISO 8601
- **Format:** `YYYY-MM-DD`
- **Examples:**
  - ✅ `2026-04-06`
  - ❌ `04/06/2026`, `6 April 2026`, `20260406`
- **Timezone:** UTC implied, use `YYYY-MM-DDTHH:MM:SSZ` if time needed
- **Unknown Dates:** Use `NULL` or `null`, never leave blank

---

## 4. ENCODING

### Default: UTF-8
- **CSV Files:** UTF-8 without BOM
- **JSON Files:** UTF-8
- **Metadata:** UTF-8 with BOM acceptable (for Excel compatibility)
- **Validation:** Ensure no mojibake or encoding artifacts

---

## 5. DATA TYPES & FORMATTING

### Image Metadata
- **authentic (label):** Binary integer (0 or 1), NOT string
- **tampering_type:** Categorical, standardized values only:
  - `authentic` (no manipulation)
  - `copy_move` (copy-move forgery)
  - `splicing` (splicing/compositing)
  - `inpainting` (region filling)
  - `gan` (GAN-generated)
  - `synthesis` (synthetic generation)
  - `text_insertion` (text overlay)
  - `unknown` (unknown tampering type)
- **quality_level:** Categorical (high, medium, low) - infer from compression
- **file_format:** Standardized extension (jpg, png, tif, bmp)

### Numeric Precision
- **File sizes:** Bytes (integer), no decimals
- **Resolutions:** String format "WxH" (e.g., "512x512")
- **Counts:** Integer
- **Percentages:** Decimal (0.0 to 1.0 range), NOT 0-100

---

## 6. MISSING DATA POLICY

### Handling Strategy by Column
| Column | Missing Handling | Reason |
|--------|-----------------|--------|
| `image_path` | DELETE ROW | Critical identifier |
| `authentic` | FLAG & INVESTIGATE | Critical label |
| `source_dataset` | INFER from path | Usually inferable |
| `tampering_type` | Set to `unknown` | Acceptable if inference impossible |
| `acquisition_date` | Set to `NULL` | Often unavailable |
| `quality_level` | INFER | Can estimate from file size/format |
| `resolution` | INFER | Can extract from image metadata |
| `processing_notes` | Leave blank | Optional annotation |

### Flagging Strategy
- Create separate `_flags.csv` for rows with missing critical data
- Format: `image_path, flag_type, resolution_method`
- Examples: `casia2_authentic_0001.jpg, missing_label, null`

---

## 7. DUPLICATE HANDLING

### File Duplicates
- **Detection:** Compare file hash (MD5 or SHA256)
- **Action:** Keep 1 copy, log others as duplicates
- **Backup:** Archive duplicate list before deletion

### Row Duplicates (Metadata)
- **Detection:** Identical `image_path` + `source_dataset`
- **Action:** Keep first occurrence, log as duplicate
- **Handling:** Merge metadata if different columns have data

### Duplicate Images (Same Visual Content)
- **Detection:** If required, use image comparison (perceptual hash)
- **Action:** NO automatic deletion, FLAG for human review
- **Documentation:** Create `visual_duplicates.csv` for analysis

---

## 8. VALIDATION RULES

### Image File Validation
- ✅ File exists and is readable
- ✅ File extension matches actual format
- ✅ File size > 0 bytes
- ✅ Image is valid and loadable (can be decoded)

### Metadata Validation
- ✅ Every image_path points to existing file
- ✅ `authentic` is 0 or 1 (not NULL)
- ✅ `tampering_type` is from standardized list
- ✅ Date format correct (if present)
- ✅ No whitespace/encoding issues in text fields

### Schema Validation
- ✅ All required columns present
- ✅ Data types match specification
- ✅ No unexpected null values in critical fields

---

## 9. DIRECTORY STRUCTURE (Output)

```
cleaned_data/
├── README.md                          (overview and usage guide)
├── TRANSFORMATION_LOG.md              (detailed change log)
├── VALIDATION_REPORT.json             (validation results)
│
├── images/                            (image files)
│   ├── casia2/
│   │   ├── authentic/
│   │   └── tampered/
│   ├── comofod/
│   │   ├── canonical_tampering/
│   │   ├── copy_move/
│   │   └── splicing/
│   ├── micc220/
│   │   ├── original/
│   │   └── tampered/
│   ├── gan_fake/
│   ├── synthetic/
│   └── training_splits/
│       ├── train/
│       ├── val/
│       └── test/
│
├── metadata/                          (CSV/JSON annotations)
│   ├── casia2_metadata.csv
│   ├── comofod_metadata.csv
│   ├── micc220_metadata.csv
│   ├── unified_groundtruth.csv        (master metadata file)
│   ├── duplicate_files.csv            (removed duplicates log)
│   └── data_quality_flags.csv         (flagged issues)
│
├── mappings/                          (reference/lookup tables)
│   ├── tampering_type_codes.json      (enum definitions)
│   ├── source_dataset_codes.json
│   └── filename_mapping.csv           (original → standardized)
│
└── logs/                              (processing logs)
    ├── processing_log.txt             (execution timeline)
    ├── errors.log                     (any issues encountered)
    └── statistics.json                (pre/post stats)
```

---

## 10. INTEGRITY CHECKLIST

Before considering data "cleaned":

- [ ] **File Count:** Documented original and final counts
- [ ] **Duplicates:** All removed or flagged
- [ ] **Extensions:** Consistent (jpg only for JPEG)
- [ ] **Naming:** All files follow convention
- [ ] **Metadata:** All images have groundtruth row
- [ ] **Validation:** All rows pass validation rules
- [ ] **Gaps:** Missing data documented and handled
- [ ] **Hashes:** MD5 hashes computed for verification
- [ ] **Documentation:** Full transformation log complete
- [ ] **Original Data:** Untouched and preserved

---

## 11. REPRODUCIBILITY

### Requirements for Reproducing Cleaning
1. **Script:** `clean_dataset.py` with all operations
2. **Log:** `TRANSFORMATION_LOG.md` with every change
3. **Config:** Standardization rules (this document)
4. **Checksums:** MD5 of original and cleaned data
5. **Version:** All software versions documented

### Revert Capability
- Original DATA/ folder never modified
- Cleaned data in cleaned_data/ (separate)
- Can safely delete cleaned_data/ and re-run if needed

---

## 12. SPECIAL CASES & EXCEPTIONS

### YOLO Dataset
- **Path Hardcoding:** Will update from hardcoded to relative path
- **Task Mismatch:** This is object detection (not binary classification)
- **Handling:** Separate from main binary classification datasets
- **Output:** Keep in distinct `training/yolo_specialized/` folder

### Layer Variants (layer4_tiny, layer4_tiny_root)
- **Decision:** These are working subsets, archive if not actively used
- **Documentation:** Create README explaining relationship to full dataset
- **Action:** Link to full dataset groundtruth, don't duplicate metadata

### .zip Files
- **Action:** Investigate before deleting
- **Backup:** Create archive of contents before removal
- **Documentation:** Log what was in each ZIP

---

## 13. QUALITY METRICS (Post-Cleaning)

Track these metrics for validation:

```json
{
  "metrics": {
    "total_images": 825000,
    "duplicates_removed": 5000,
    "garbage_files_removed": 31,
    "extension_standardization": {
      "jpeg_to_jpg_converted": 77336,
      "consistency_score": 1.0
    },
    "metadata_coverage": {
      "images_with_groundtruth": 825000,
      "coverage_percentage": 100
    },
    "validation": {
      "files_passed_checks": 825000,
      "files_with_issues": 0
    },
    "disk_space": {
      "original_gb": 17.5,
      "cleaned_gb": 12.3,
      "saved_gb": 5.2
    }
  }
}
```

---

## Document Approval

**Status:** Ready for Implementation  
**Last Updated:** 2026-04-06  
**Author:** Data Standardization Process

