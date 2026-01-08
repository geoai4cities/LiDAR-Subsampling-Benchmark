#!/usr/bin/env python3
"""
Find good scans for point cloud visualization figures.

Scans all sequences to find scans with good building and vegetation content.
Saves results to a text file for use by generate_pointcloud_comparison.py

For Response to Reviewers - Comment 6.14
"""

import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Label mapping for SemanticKITTI (from original labels to training labels)
LABEL_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5,
    30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13,
    51: 14, 52: 0, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 19,
    99: 0, 252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5,
}


def load_labels(label_path):
    """Load semantic labels from label file."""
    labels = np.fromfile(label_path, dtype=np.uint32)
    sem_labels = labels & 0xFFFF  # Lower 16 bits are semantic labels
    # Map to training labels
    mapped_labels = np.vectorize(lambda x: LABEL_MAP.get(x, 0))(sem_labels)
    return mapped_labels


def analyze_scan(label_path):
    """Analyze a scan for building and vegetation content."""
    labels = load_labels(label_path)
    total = len(labels)

    building_count = np.sum(labels == 13)  # building class
    veg_count = np.sum(labels == 15)       # vegetation class
    road_count = np.sum(labels == 9)       # road class
    car_count = np.sum(labels == 1)        # car class

    return {
        'total': total,
        'building': building_count,
        'vegetation': veg_count,
        'road': road_count,
        'car': car_count,
        'building_pct': building_count / total * 100 if total > 0 else 0,
        'veg_pct': veg_count / total * 100 if total > 0 else 0,
    }


def main():
    """Main function to find good scans."""

    # Paths
    original_dir = PROJECT_ROOT / 'data' / 'SemanticKITTI' / 'original' / 'sequences'
    output_file = PROJECT_ROOT / 'scripts' / 'figures' / 'good_scans.txt'

    # Sequences to scan (all training + validation sequences)
    sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

    print("="*70)
    print("Finding good scans for point cloud visualization")
    print("="*70)

    all_results = []

    for seq in sequences:
        labels_dir = original_dir / seq / 'labels'

        if not labels_dir.exists():
            print(f"Sequence {seq}: NOT FOUND")
            continue

        label_files = sorted(labels_dir.glob('*.label'))
        print(f"\nSequence {seq}: {len(label_files)} scans")
        print("-"*50)

        seq_results = []

        for i, label_file in enumerate(label_files):
            scan_id = label_file.stem
            stats = analyze_scan(label_file)

            # Score based on building and vegetation presence
            # We want scans with substantial amounts of both
            score = min(stats['building'], 3000) + min(stats['vegetation'], 3000)

            # Add bonus for having both classes
            if stats['building'] > 500 and stats['vegetation'] > 500:
                score += 2000

            # Add bonus for having road (provides context)
            if stats['road'] > 1000:
                score += 500

            stats['scan_id'] = scan_id
            stats['sequence'] = seq
            stats['score'] = score
            seq_results.append(stats)

            # Progress indicator
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(label_files)} scans...")

        # Sort by score (descending)
        seq_results.sort(key=lambda x: x['score'], reverse=True)

        # Show top 10 for this sequence
        print(f"\n  Top 10 scans for sequence {seq}:")
        print(f"  {'Scan':<10} {'Score':<8} {'Building':<10} {'Veg':<10} {'Road':<10}")
        print(f"  {'-'*48}")

        for result in seq_results[:10]:
            print(f"  {result['scan_id']:<10} {result['score']:<8.0f} "
                  f"{result['building']:<10} {result['vegetation']:<10} {result['road']:<10}")

        all_results.extend(seq_results[:20])  # Keep top 20 from each sequence

    # Overall best scans
    all_results.sort(key=lambda x: x['score'], reverse=True)

    print("\n" + "="*70)
    print("OVERALL TOP SCANS FOR VISUALIZATION")
    print("="*70)

    # Categorize best scans
    best_building = max(all_results, key=lambda x: x['building'])
    best_vegetation = max(all_results, key=lambda x: x['vegetation'])
    best_balanced = max(all_results, key=lambda x: min(x['building'], x['vegetation']))

    print(f"\nBest for BUILDING visualization:")
    print(f"  Sequence {best_building['sequence']}, Scan {best_building['scan_id']}")
    print(f"  Building: {best_building['building']:,} points ({best_building['building_pct']:.1f}%)")
    print(f"  Vegetation: {best_building['vegetation']:,} points")

    print(f"\nBest for VEGETATION visualization:")
    print(f"  Sequence {best_vegetation['sequence']}, Scan {best_vegetation['scan_id']}")
    print(f"  Vegetation: {best_vegetation['vegetation']:,} points ({best_vegetation['veg_pct']:.1f}%)")
    print(f"  Building: {best_vegetation['building']:,} points")

    print(f"\nBest BALANCED (both building & vegetation):")
    print(f"  Sequence {best_balanced['sequence']}, Scan {best_balanced['scan_id']}")
    print(f"  Building: {best_balanced['building']:,} points")
    print(f"  Vegetation: {best_balanced['vegetation']:,} points")

    # Save results to file
    with open(output_file, 'w') as f:
        f.write("# Good scans for point cloud visualization\n")
        f.write("# Generated by find_good_scans.py\n")
        f.write("# Format: sequence,scan_id,score,building_count,vegetation_count\n")
        f.write("#\n")
        f.write(f"# Best for BUILDING: seq={best_building['sequence']}, scan={best_building['scan_id']}\n")
        f.write(f"# Best for VEGETATION: seq={best_vegetation['sequence']}, scan={best_vegetation['scan_id']}\n")
        f.write(f"# Best BALANCED: seq={best_balanced['sequence']}, scan={best_balanced['scan_id']}\n")
        f.write("#\n")
        f.write("# RECOMMENDED SCANS:\n")
        f.write(f"building_scan={best_building['sequence']},{best_building['scan_id']}\n")
        f.write(f"vegetation_scan={best_vegetation['sequence']},{best_vegetation['scan_id']}\n")
        f.write(f"balanced_scan={best_balanced['sequence']},{best_balanced['scan_id']}\n")
        f.write("#\n")
        f.write("# All top scans (sorted by score):\n")

        for result in all_results[:30]:
            f.write(f"{result['sequence']},{result['scan_id']},{result['score']:.0f},"
                   f"{result['building']},{result['vegetation']}\n")

    print(f"\nResults saved to: {output_file}")
    print("="*70)


if __name__ == '__main__':
    main()
