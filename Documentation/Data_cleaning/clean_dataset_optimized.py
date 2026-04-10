#!/usr/bin/env python3
"""
VeriSight Data Cleaning Pipeline - OPTIMIZED
Focus on critical cleaning operations without file hashing
"""

import os
import sys
import json
import csv
import shutil
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = Path('Data')
CLEANED_DIR = Path('cleaned_data')

class OptimizedCleaningPipeline:
    """Optimized pipeline for critical cleaning operations"""

    def __init__(self):
        self.stats = {
            'garbage_removed': 0,
            'jpeg_converted': 0,
            'metadata_created': 0
        }
        self.transformations = []

    def log_transform(self, action, details):
        entry = {'time': datetime.now().isoformat(), 'action': action, 'details': details}
        self.transformations.append(entry)
        logger.info(f"{action}: {details}")

    def setup_structure(self):
        """Create cleaned_data directory"""
        logger.info("\n[STEP 1] Creating directory structure...")
        base_dirs = [
            'images/casia2/authentic',
            'images/casia2/tampered',
            'images/micc220/original',
            'images/micc220/tampered',
            'metadata',
            'logs'
        ]
        for subdir in base_dirs:
            (CLEANED_DIR / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created cleaned_data/ structure")

    def stage_step2_jpeg_conversion(self):
        """Plan JPEG -> JPG conversion (list files, don't execute yet)"""
        logger.info("\n[STEP 2-PRE] Identifying .JPEG files to convert...")

        jpeg_files = list(DATA_DIR.rglob('*.JPEG'))
        logger.info(f"Found {len(jpeg_files)} .JPEG files requiring conversion")

        conversion_plan = []
        for jpeg_file in jpeg_files:
            jpg_file = jpeg_file.with_suffix('.jpg')
            conversion_plan.append({'from': str(jpeg_file), 'to': str(jpg_file)})

        # Save plan
        plan_file = CLEANED_DIR / 'logs' / 'jpeg_conversion_plan.json'
        with open(plan_file, 'w') as f:
            json.dump(conversion_plan, f, indent=2)

        logger.info(f"Saved conversion plan: {plan_file}")
        return len(jpeg_files)

    def step2_remove_garbage(self):
        """Remove garbage files"""
        logger.info("\n[STEP 2] Removing garbage files...")

        garbage_patterns = ['.DS_Store', '.cache', '.done']
        count = 0

        for root, dirs, files in os.walk(DATA_DIR):
            for file in files:
                for pattern in garbage_patterns:
                    if file.endswith(pattern):
                        filepath = Path(root) / file
                        try:
                            filepath.unlink()
                            self.log_transform('REMOVED', str(filepath))
                            count += 1
                        except Exception as e:
                            logger.error(f"Error removing {filepath}: {e}")

        self.stats['garbage_removed'] = count
        logger.info(f"Removed {count} garbage files")

    def step3_standardize_jpeg(self):
        """Convert .JPEG to .jpg"""
        logger.info("\n[STEP 3] Standardizing .JPEG -> .jpg...")

        jpeg_files = list(DATA_DIR.rglob('*.JPEG'))
        count = 0

        for jpeg_file in jpeg_files:
            jpg_file = jpeg_file.with_suffix('.jpg')
            try:
                jpeg_file.rename(jpg_file)
                self.log_transform('RENAMED', f"{jpeg_file.name} -> {jpg_file.name}")
                count += 1
                if count % 1000 == 0:
                    logger.info(f"  Converted {count}/{len(jpeg_files)}")
            except Exception as e:
                logger.error(f"Error converting {jpeg_file}: {e}")

        self.stats['jpeg_converted'] = count
        logger.info(f"Converted {count} files from .JPEG to .jpg")

    def step4_create_metadata(self):
        """Create metadata for key datasets"""
        logger.info("\n[STEP 4] Creating unified metadata...")

        all_records = []

        # CASIA2 metadata
        casia2_au = DATA_DIR / 'CASIA2' / 'Au'
        if casia2_au.exists():
            for img in casia2_au.glob('*.jpg'):
                all_records.append({
                    'image_path': f'casia2/authentic/{img.name}',
                    'source_dataset': 'CASIA2',
                    'authentic': 1,
                    'tampering_type': 'none',
                    'file_format': 'jpg',
                    'file_size_bytes': img.stat().st_size
                })
        logger.info(f"Processed CASIA2 authentic images")

        casia2_tp = DATA_DIR / 'CASIA2' / 'Tp'
        if casia2_tp.exists():
            for img in casia2_tp.glob('*.jpg'):
                all_records.append({
                    'image_path': f'casia2/tampered/{img.name}',
                    'source_dataset': 'CASIA2',
                    'authentic': 0,
                    'tampering_type': 'unknown',
                    'file_format': 'jpg',
                    'file_size_bytes': img.stat().st_size
                })
        logger.info(f"Processed CASIA2 tampered images")

        # MICC-F220 metadata from groundtruth
        micc_gt = DATA_DIR / 'MICC-F220' / 'groundtruthDB_220.txt'
        if micc_gt.exists():
            with open(micc_gt, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = int(parts[1])

                        if 'tamp' in filename:
                            img_type = 'tampered'
                            authentic = 0
                        else:
                            img_type = 'original'
                            authentic = 1

                        all_records.append({
                            'image_path': f'micc220/{img_type}/{filename}',
                            'source_dataset': 'MICC-F220',
                            'authentic': authentic,
                            'tampering_type': 'unknown' if authentic == 0 else 'none',
                            'file_format': 'jpg',
                            'file_size_bytes': 0
                        })
        logger.info(f"Processed MICC-F220 from groundtruth")

        # Write unified groundtruth
        output_file = CLEANED_DIR / 'metadata' / 'unified_groundtruth.csv'
        fieldnames = ['image_path', 'source_dataset', 'authentic', 'tampering_type', 'file_format', 'file_size_bytes']

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_records)

        self.stats['metadata_created'] = len(all_records)
        logger.info(f"Created unified groundtruth: {len(all_records)} entries")

    def step5_summarize(self):
        """Generate final summary"""
        logger.info("\n" + "="*60)
        logger.info("DATA CLEANING PIPELINE SUMMARY")
        logger.info("="*60)
        logger.info(f"Garbage files removed:  {self.stats['garbage_removed']}")
        logger.info(f"JPEG files converted:   {self.stats['jpeg_converted']}")
        logger.info(f"Metadata entries:       {self.stats['metadata_created']}")
        logger.info("="*60)

        # Save summary
        summary_file = CLEANED_DIR / 'logs' / 'cleaning_summary.json'
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'stats': self.stats,
                'transformations_count': len(self.transformations)
            }, f, indent=2)

        # Save transformations log
        log_file = CLEANED_DIR / 'logs' / 'transformations.json'
        with open(log_file, 'w') as f:
            json.dump(self.transformations, f, indent=2)

        logger.info(f"\nResults saved to cleaned_data/logs/")

    def run(self):
        """Execute pipeline"""
        logger.info("\n" + "="*60)
        logger.info("VeriSight OPTIMIZED CLEANING PIPELINE")
        logger.info("="*60)

        start = datetime.now()

        self.setup_structure()
        jpeg_count = self.stage_step2_jpeg_conversion()
        self.step2_remove_garbage()
        self.step3_standardize_jpeg()
        self.step4_create_metadata()
        self.step5_summarize()

        elapsed = (datetime.now() - start).total_seconds()
        logger.info(f"\nCompleted in {elapsed:.1f} seconds")

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    pipeline = OptimizedCleaningPipeline()
    pipeline.run()
