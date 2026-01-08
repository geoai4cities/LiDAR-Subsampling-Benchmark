"""
Convert DALES PLY files to TXT format for PTv3 training

DALES dataset comes in PLY format but the PTv3 dataset loader expects TXT format.
This script converts PLY files to TXT with the required format:
x, y, z, intensity, return_num, num_returns, class

Input: data/DALES/original/dales_ply/{train,test}/*.ply
Output: data/DALES/original/{train,test}/*.txt

Usage:
    python scripts/convert_ply_to_txt.py
    python scripts/convert_ply_to_txt.py --input data/DALES/original/dales_ply --output data/DALES/original
    python scripts/convert_ply_to_txt.py --workers 16  # Parallel processing
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

try:
    from plyfile import PlyData
except ImportError:
    print("ERROR: plyfile not installed. Install with: pip install plyfile")
    sys.exit(1)


def read_ply(file_path):
    """
    Read PLY file and extract point cloud data.

    Args:
        file_path: Path to .ply file

    Returns:
        dict with keys: x, y, z, intensity, return_num, num_returns, class
    """
    plydata = PlyData.read(file_path)
    vertex = plydata['vertex']

    # Extract data
    data = {
        'x': np.array(vertex['x']),
        'y': np.array(vertex['y']),
        'z': np.array(vertex['z']),
    }

    # Check for intensity (may have different names)
    if 'intensity' in vertex.data.dtype.names:
        data['intensity'] = np.array(vertex['intensity'])
    elif 'scalar_Intensity' in vertex.data.dtype.names:
        data['intensity'] = np.array(vertex['scalar_Intensity'])
    elif 'Intensity' in vertex.data.dtype.names:
        data['intensity'] = np.array(vertex['Intensity'])
    else:
        data['intensity'] = np.zeros(len(data['x']))

    # Check for return numbers
    if 'return_num' in vertex.data.dtype.names:
        data['return_num'] = np.array(vertex['return_num'])
    elif 'return_number' in vertex.data.dtype.names:
        data['return_num'] = np.array(vertex['return_number'])
    elif 'return' in vertex.data.dtype.names:
        data['return_num'] = np.array(vertex['return'])
    else:
        data['return_num'] = np.ones(len(data['x']), dtype=np.int32)

    # Check for number of returns
    if 'num_returns' in vertex.data.dtype.names:
        data['num_returns'] = np.array(vertex['num_returns'])
    elif 'number_of_returns' in vertex.data.dtype.names:
        data['num_returns'] = np.array(vertex['number_of_returns'])
    else:
        data['num_returns'] = np.ones(len(data['x']), dtype=np.int32)

    # Check for classification (label)
    if 'classification' in vertex.data.dtype.names:
        data['class'] = np.array(vertex['classification'])
    elif 'label' in vertex.data.dtype.names:
        data['class'] = np.array(vertex['label'])
    elif 'class' in vertex.data.dtype.names:
        data['class'] = np.array(vertex['class'])
    elif 'scalar_Classification' in vertex.data.dtype.names:
        data['class'] = np.array(vertex['scalar_Classification'])
    else:
        raise ValueError(f"No classification field found in {file_path}")

    return data


def write_txt(data, file_path):
    """
    Write point cloud data to TXT file in PTv3 format.

    Format: x, y, z, intensity, return_num, num_returns, class
    """
    # Stack all columns
    output = np.column_stack([
        data['x'],
        data['y'],
        data['z'],
        data['intensity'],
        data['return_num'],
        data['num_returns'],
        data['class']
    ])

    # Save with space delimiter, no header
    np.savetxt(file_path, output, fmt='%.6f %.6f %.6f %.3f %d %d %d')


def convert_single_file(args):
    """
    Convert a single PLY file to TXT format.

    Args:
        args: tuple of (ply_file, output_dir)

    Returns:
        tuple of (filename, n_points, success)
    """
    ply_file, output_dir = args
    try:
        # Read PLY
        data = read_ply(ply_file)

        # Write TXT with same name
        txt_file = output_dir / (ply_file.stem + '.txt')
        write_txt(data, txt_file)

        return (ply_file.name, len(data['x']), True)
    except Exception as e:
        return (ply_file.name, 0, False, str(e))


def convert_ply_to_txt_parallel(input_dir, output_dir, split='train', workers=None, verbose=True):
    """
    Convert all PLY files in a directory to TXT format using parallel processing.

    Args:
        input_dir: Input directory containing .ply files
        output_dir: Output directory for .txt files
        split: 'train' or 'test'
        workers: Number of parallel workers (default: 75% of CPU cores)
        verbose: Print progress
    """
    input_path = Path(input_dir) / split
    output_path = Path(output_dir) / split

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all PLY files
    ply_files = sorted(input_path.glob('*.ply'))

    if len(ply_files) == 0:
        print(f"WARNING: No PLY files found in {input_path}")
        return

    # Determine number of workers
    if workers is None:
        workers = max(1, int(cpu_count() * 0.75))
    workers = min(workers, len(ply_files))

    if verbose:
        print(f"\nConverting {len(ply_files)} PLY files using {workers} workers")
        print(f"  From: {input_path}")
        print(f"  To:   {output_path}")

    # Prepare arguments for parallel processing
    task_args = [(ply_file, output_path) for ply_file in ply_files]

    # Process in parallel with progress bar
    results = []
    with Pool(workers) as pool:
        for result in tqdm(
            pool.imap_unordered(convert_single_file, task_args),
            total=len(ply_files),
            desc=f"Converting {split}",
            disable=not verbose
        ):
            results.append(result)

    # Check results
    success_count = sum(1 for r in results if r[2])
    fail_count = len(results) - success_count
    total_points = sum(r[1] for r in results if r[2])

    if verbose:
        print(f"✓ Converted {success_count}/{len(ply_files)} files ({total_points:,} total points)")
        if fail_count > 0:
            print(f"✗ {fail_count} files failed:")
            for r in results:
                if not r[2]:
                    print(f"  - {r[0]}: {r[3] if len(r) > 3 else 'Unknown error'}")


def verify_conversion(txt_file, expected_cols=7):
    """
    Verify that a TXT file has the correct format.
    """
    try:
        data = np.loadtxt(txt_file)

        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {data.shape}")

        if data.shape[1] != expected_cols:
            raise ValueError(f"Expected {expected_cols} columns, got {data.shape[1]}")

        return True

    except Exception as e:
        print(f"ERROR verifying {txt_file}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert DALES PLY files to TXT format for PTv3 training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with default paths (parallel)
  python scripts/preprocessing/convert_ply_to_txt.py

  # Convert with custom paths
  python scripts/preprocessing/convert_ply_to_txt.py \\
    --input data/DALES/original/dales_ply \\
    --output data/DALES/original

  # Use specific number of workers
  python scripts/preprocessing/convert_ply_to_txt.py --workers 16

  # Verify conversion
  python scripts/preprocessing/convert_ply_to_txt.py --verify-only
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        default='data/DALES/original/dales_ply',
        help='Input directory containing PLY files (default: data/DALES/original/dales_ply)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/DALES/original',
        help='Output directory for TXT files (default: data/DALES/original)'
    )

    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'test'],
        choices=['train', 'test'],
        help='Which splits to convert (default: train test)'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        help='Number of parallel workers (default: 75%% of CPU cores)'
    )

    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing TXT files, do not convert'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Get absolute paths
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    # Determine workers
    if args.workers is None:
        workers = max(1, int(cpu_count() * 0.75))
    else:
        workers = args.workers

    print("=" * 80)
    print("DALES PLY to TXT Converter (Parallel)")
    print("=" * 80)
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Splits:  {args.splits}")
    print(f"Workers: {workers}")
    print("=" * 80)

    if args.verify_only:
        # Verify existing TXT files
        print("\nVerifying existing TXT files...")
        for split in args.splits:
            txt_path = output_dir / split
            if not txt_path.exists():
                print(f"WARNING: {txt_path} does not exist")
                continue

            txt_files = sorted(txt_path.glob('*.txt'))
            print(f"\nVerifying {len(txt_files)} files in {split} split...")

            failed = 0
            for txt_file in tqdm(txt_files, desc=f"Verifying {split}"):
                if not verify_conversion(txt_file):
                    failed += 1

            if failed == 0:
                print(f"✓ All {len(txt_files)} files verified for {split} split")
            else:
                print(f"✗ {failed}/{len(txt_files)} files failed verification for {split} split")

    else:
        # Convert PLY to TXT in parallel
        for split in args.splits:
            convert_ply_to_txt_parallel(
                input_dir,
                output_dir,
                split=split,
                workers=workers,
                verbose=not args.quiet
            )

        # Verify conversion
        if not args.quiet:
            print("\n" + "=" * 80)
            print("Verifying conversion...")
            print("=" * 80)

            for split in args.splits:
                txt_path = output_dir / split
                txt_files = sorted(txt_path.glob('*.txt'))

                if len(txt_files) > 0:
                    # Verify first and last file
                    print(f"\nVerifying {split} split (sample)...")
                    verify_conversion(txt_files[0])
                    if len(txt_files) > 1:
                        verify_conversion(txt_files[-1])
                    print(f"✓ Sample verification passed for {split} split")

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
