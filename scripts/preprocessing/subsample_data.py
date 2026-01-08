#!/usr/bin/env python
"""
Subsample datasets using specified method
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'src'))

from subsampling import *

def main():
    parser = argparse.ArgumentParser(description='Subsample point cloud data')
    parser.add_argument('--dataset', type=str, required=True, choices=['DALES', 'SemanticKITTI'])
    parser.add_argument('--method', type=str, required=True, 
                       choices=['RS', 'SB', 'VB', 'IDIS', 'DEPOCO', 'FPS', 'DBSCAN'])
    parser.add_argument('--loss_level', type=str, required=True,
                       help='Loss level, e.g., 50-55')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print(f"Subsampling {args.dataset} using {args.method} at {args.loss_level}% loss")
    
    # TODO: Implement subsampling logic
    # 1. Load data
    # 2. Apply subsampling method
    # 3. Save subsampled data
    
    print("Subsampling complete!")

if __name__ == '__main__':
    main()
