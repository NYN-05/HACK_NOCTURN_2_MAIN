#!/usr/bin/env python3
"""
VeriSight Data Cleaning and Standardization Pipeline
Transforms raw data directory into clean, standardized format
Maintains original data, outputs to cleaned_data/
"""

import os
import sys
import json
import csv
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cleaning_process.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
DATA_DIR = Path('Data')
CLEANED_DIR = Path('cleaned_data')
GARBAGE_EXTENSIONS = {'.DS_Store', '.cache', '.done'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

class DataCleaningPipeline:
    """Main pipeline for data cleaning and standardization"""

    def __init__(self):
        self.stats = {
            'garbage_files_removed': 0,
            'duplicates_found': 0,
            'extension_conversions': 0,
            'metadata_entries_created': 0,
            'errors': []
        }
        self.file_hashes = {}  # For duplicate detection
        self.metadata_records = defaultdict(list)
        self.transformation_log = []
        self.duplicate_files = []

    def log_transformation(self, source, action, notes=""):
        """Record transformation in log"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'action': action,
            'notes': notes
        }
        self.transformation_log.append(entry)
        logger.info(f"[{action}] {source}: {notes}")

    def compute_hash(self, filepath, algorithm='md5'):
        """Compute file hash for duplicate detection"""
        hash_obj = hashlib.new(algorithm)
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return None

    def step1_create_output_structure(self):
        """Create cleaned_data directory structure"""
        logger.info("=" * 60)
        logger.info("STEP 1: Creating output directory structure")
        logger.info("=" * 60)

        dirs_to_create = [
            CLEANED_DIR / 'images' / 'casia2' / 'authentic',
            CLEANED_DIR / 'images' / 'casia2' / 'tampered',
            CLEANED_DIR / 'images' / 'comofod',
            CLEANED_DIR / 'images' / 'micc220' / 'original',
            CLEANED_DIR / 'images' / 'micc220' / 'tampered',
            CLEANED_DIR / 'images' / 'gan_fake',
            CLEANED_DIR / 'images' / 'synthetic',
            CLEANED_DIR / 'images' / 'training_splits' / 'train' / 'authentic',
            CLEANED_DIR / 'images' / 'training_splits' / 'train' / 'manipulated',
            CLEANED_DIR / 'images' / 'training_splits' / 'val' / 'authentic',
            CLEANED_DIR / 'images' / 'training_splits' / 'val' / 'manipulated',
            CLEANED_DIR / 'images' / 'training_splits' / 'test' / 'authentic',
            CLEANED_DIR / 'images' / 'training_splits' / 'test' / 'manipulated',
            CLEANED_DIR / 'metadata',
            CLEANED_DIR / 'mappings',
            CLEANED_DIR / 'logs'
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {dir_path}")

        logger.info(f"Structure created successfully under {CLEANED_DIR}")

    def step2_remove_garbage_files(self):
        """Remove system files and artifacts"""
        logger.info("=" * 60)
        logger.info("STEP 2: Removing garbage files")
        logger.info("=" * 60)

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                filepath = Path(root) / file
                if filepath.suffix in GARBAGE_EXTENSIONS:
                    try:
                        filepath.unlink()
                        self.log_transformation(str(filepath), 'DELETED', 'Garbage file')
                        self.stats['garbage_files_removed'] += 1
                    except Exception as e:
                        logger.error(f"Error removing {filepath}: {e}")
                        self.stats['errors'].append(str(e))

        logger.info(f"Removed {self.stats['garbage_files_removed']} garbage files")

    def step3_detect_duplicates(self):
        """Find duplicate files using hash"""
        logger.info("=" * 60)
        logger.info("STEP 3: Detecting duplicate files")
        logger.info("=" * 60)

        hash_to_files = defaultdict(list)

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                filepath = Path(root) / file
                if filepath.suffix.lower() in IMAGE_EXTENSIONS:
                    filehash = self.compute_hash(filepath)
                    if filehash:
                        hash_to_files[filehash].append(filepath)

        # Identify duplicates
        duplicates_count = 0
        for filehash, filepaths in hash_to_files.items():
            if len(filepaths) > 1:
                duplicates_count += len(filepaths) - 1
                # Log all but first as duplicates
                for dup in filepaths[1:]:
                    self.duplicate_files.append({
                        'original': str(filepaths[0]),
                        'duplicate': str(dup),
                        'hash': filehash
                    })
                    logger.warning(f"Duplicate: {dup} (same as {filepaths[0]})")

        self.stats['duplicates_found'] = duplicates_count
        logger.info(f"Found {duplicates_count} duplicate files")

    def step4_standardize_extensions(self):
        """Convert .JPEG to .jpg consistently"""
        logger.info("=" * 60)
        logger.info("STEP 4: Standardizing file extensions")
        logger.info("=" * 60)

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                filepath = Path(root) / file
                if filepath.suffix.upper() == '.JPEG':
                    new_filepath = filepath.with_suffix('.jpg')
                    try:
                        filepath.rename(new_filepath)
                        self.log_transformation(str(filepath), 'RENAMED',
                                              f'.JPEG -> .jpg')
                        self.stats['extension_conversions'] += 1
                    except Exception as e:
                        logger.error(f"Error renaming {filepath}: {e}")
                        self.stats['errors'].append(str(e))

        logger.info(f"Standardized {self.stats['extension_conversions']} files")

    def step5_create_metadata_casia2(self):
        """Create unified metadata for CASIA2"""
        logger.info("=" * 60)
        logger.info("STEP 5a: Creating CASIA2 metadata")
        logger.info("=" * 60)

        casia2_dir = DATA_DIR / 'CASIA2'
        if not casia2_dir.exists():
            logger.warning("CASIA2 directory not found")
            return

        records = []

        # Authentic images
        au_dir = casia2_dir / 'Au'
        if au_dir.exists():
            for img_file in au_dir.glob('*.jpg'):
                try:
                    record = {
                        'image_path': f'casia2/authentic/{img_file.name}',
                        'source_dataset': 'CASIA2',
                        'filename_original': img_file.name,
                        'authentic': 1,
                        'tampering_type': 'authentic',
                        'quality_level': 'high',
                        'resolution': 'unknown',
                        'file_format': 'jpg',
                        'file_size_bytes': img_file.stat().st_size,
                        'acquisition_date': None,
                        'processing_notes': 'Original CASIA2 authentic image'
                    }
                    records.append(record)
                    self.stats['metadata_entries_created'] += 1
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")

        # Tampered images
        tp_dir = casia2_dir / 'Tp'
        if tp_dir.exists():
            for img_file in tp_dir.glob('*.jpg'):
                try:
                    record = {
                        'image_path': f'casia2/tampered/{img_file.name}',
                        'source_dataset': 'CASIA2',
                        'filename_original': img_file.name,
                        'authentic': 0,
                        'tampering_type': 'unknown',
                        'quality_level': 'high',
                        'resolution': 'unknown',
                        'file_format': 'jpg',
                        'file_size_bytes': img_file.stat().st_size,
                        'acquisition_date': None,
                        'processing_notes': 'Original CASIA2 tampered image'
                    }
                    records.append(record)
                    self.stats['metadata_entries_created'] += 1
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")

        self.metadata_records['casia2'] = records
        logger.info(f"Created {len(records)} CASIA2 metadata entries")

    def step5b_create_metadata_micc220(self):
        """Create metadata from MICC-F220 groundtruth"""
        logger.info("=" * 60)
        logger.info("STEP 5b: Creating MICC-F220 metadata")
        logger.info("=" * 60)

        micc_dir = DATA_DIR / 'MICC-F220'
        gt_file = micc_dir / 'groundtruthDB_220.txt'

        if not gt_file.exists():
            logger.warning(f"MICC groundtruth file not found: {gt_file}")
            return

        records = []
        try:
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = int(parts[1])

                        # Determine authentic/tampered from naming
                        if 'tamp' in filename:
                            authentic = 0
                            tampering_type = 'unknown'
                            image_type = 'tampered'
                        else:
                            authentic = 1
                            tampering_type = 'authentic'
                            image_type = 'original'

                        record = {
                            'image_path': f'micc220/{image_type}/{filename}',
                            'source_dataset': 'MICC-F220',
                            'filename_original': filename,
                            'authentic': authentic,
                            'tampering_type': tampering_type,
                            'quality_level': 'high',
                            'resolution': 'unknown',
                            'file_format': 'jpg',
                            'file_size_bytes': 0,  # Will update if file exists
                            'acquisition_date': None,
                            'processing_notes': f'Label: {label}'
                        }

                        # Try to get actual file size
                        img_path = micc_dir / filename
                        if img_path.exists():
                            record['file_size_bytes'] = img_path.stat().st_size

                        records.append(record)
                        self.stats['metadata_entries_created'] += 1

        except Exception as e:
            logger.error(f"Error reading MICC groundtruth: {e}")

        self.metadata_records['micc220'] = records
        logger.info(f"Created {len(records)} MICC-F220 metadata entries")

    def step5c_create_unified_groundtruth(self):
        """Merge all metadata into unified groundtruth CSV"""
        logger.info("=" * 60)
        logger.info("STEP 5c: Creating unified groundtruth file")
        logger.info("=" * 60)

        all_records = []
        for source, records in self.metadata_records.items():
            all_records.extend(records)

        # Write unified CSV
        output_file = CLEANED_DIR / 'metadata' / 'unified_groundtruth.csv'
        fieldnames = [
            'image_path', 'source_dataset', 'filename_original', 'authentic',
            'tampering_type', 'quality_level', 'resolution', 'file_format',
            'file_size_bytes', 'acquisition_date', 'processing_notes'
        ]

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_records)

            logger.info(f"Wrote unified groundtruth: {output_file}")
            logger.info(f"Total entries: {len(all_records)}")
        except Exception as e:
            logger.error(f"Error writing unified groundtruth: {e}")
            self.stats['errors'].append(str(e))

    def step6_write_duplicate_log(self):
        """Write duplicate files report"""
        logger.info("=" * 60)
        logger.info("STEP 6: Writing duplicate files log")
        logger.info("=" * 60)

        output_file = CLEANED_DIR / 'metadata' / 'duplicate_files.csv'

        if self.duplicate_files:
            try:
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['original', 'duplicate', 'hash'])
                    writer.writeheader()
                    writer.writerows(self.duplicate_files)

                logger.info(f"Wrote duplicate log: {output_file} ({len(self.duplicate_files)} duplicates)")
            except Exception as e:
                logger.error(f"Error writing duplicate log: {e}")
        else:
            logger.info("No duplicates found")

    def step7_write_transformation_log(self):
        """Write detailed transformation log"""
        logger.info("=" * 60)
        logger.info("STEP 7: Writing transformation log")
        logger.info("=" * 60)

        output_file = CLEANED_DIR / 'logs' / 'transformation_log.json'

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.transformation_log, f, indent=2)

            logger.info(f"Wrote transformation log: {output_file}")
        except Exception as e:
            logger.error(f"Error writing transformation log: {e}")

    def step8_write_statistics(self):
        """Write final statistics"""
        logger.info("=" * 60)
        logger.info("STEP 8: Writing statistics report")
        logger.info("=" * 60)

        output_file = CLEANED_DIR / 'logs' / 'statistics.json'

        stats_report = {
            'timestamp': datetime.now().isoformat(),
            'pipeline_version': '1.0',
            'statistics': self.stats,
            'metadata_created': self.stats['metadata_entries_created']
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(stats_report, f, indent=2)

            logger.info(f"Wrote statistics: {output_file}")
            logger.info(f"\n{'='*60}")
            logger.info("CLEANING PIPELINE SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Garbage files removed:     {self.stats['garbage_files_removed']}")
            logger.info(f"Duplicates found:          {self.stats['duplicates_found']}")
            logger.info(f"Extension conversions:     {self.stats['extension_conversions']}")
            logger.info(f"Metadata entries created:  {self.stats['metadata_entries_created']}")
            logger.info(f"Errors encountered:        {len(self.stats['errors'])}")
            if self.stats['errors']:
                for error in self.stats['errors']:
                    logger.error(f"  - {error}")
            logger.info(f"{'='*60}\n")
        except Exception as e:
            logger.error(f"Error writing statistics: {e}")

    def run_pipeline(self):
        """Execute complete cleaning pipeline"""
        logger.info("\n" + "="*60)
        logger.info("VeriSight DATA CLEANING PIPELINE START")
        logger.info("="*60 + "\n")

        start_time = datetime.now()

        try:
            self.step1_create_output_structure()
            self.step2_remove_garbage_files()
            self.step3_detect_duplicates()
            self.step4_standardize_extensions()
            self.step5_create_metadata_casia2()
            self.step5b_create_metadata_micc220()
            self.step5c_create_unified_groundtruth()
            self.step6_write_duplicate_log()
            self.step7_write_transformation_log()
            self.step8_write_statistics()

            elapsed = datetime.now() - start_time
            logger.info("\n" + "="*60)
            logger.info("VeriSight DATA CLEANING PIPELINE COMPLETE")
            logger.info(f"Elapsed time: {elapsed}")
            logger.info("="*60 + "\n")

            return True

        except Exception as e:
            logger.critical(f"Pipeline failed: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)  # Run from VERISIGHT_V1 directory
    pipeline = DataCleaningPipeline()
    success = pipeline.run_pipeline()
    sys.exit(0 if success else 1)
