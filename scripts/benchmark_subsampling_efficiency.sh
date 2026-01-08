#!/bin/bash
################################################################################
# Subsampling Efficiency Benchmark
#
# This script benchmarks all subsampling methods and measures:
#   - Wall-clock time (total and per-scan)
#   - Peak memory usage (RAM for CPU, VRAM for GPU)
#   - CPU/GPU utilization
#   - Throughput (scans/second)
#
# Runs 3 iterations and reports averaged results.
#
# Output: benchmark_results/ directory with timing and resource metrics
#
################################################################################
# Usage
################################################################################
#
#   ./benchmark_subsampling_efficiency.sh [OPTIONS]
#
#   Options:
#     --sequences SEQ     Sequences to benchmark (default: "08" for validation)
#     --loss LOSS         Loss percentage (default: 70)
#     --output-dir DIR    Output directory for benchmark results
#     --skip-existing     Skip methods that already have benchmark results
#     --methods METHODS   Comma-separated list of methods to benchmark
#                         (default: all - RS,DBSCAN,Voxel,Poisson,IDIS,FPS,DEPOCO)
#     --iterations N      Number of iterations to run (default: 3)
#
################################################################################
# Metrics Collected
################################################################################
#
# Time Metrics:
#   - Total wall-clock time (seconds)
#   - Time per scan (seconds)
#   - Throughput (scans/second)
#
# Memory Metrics:
#   - Peak RAM usage (GB) - for all methods
#   - Peak GPU memory (GB) - for GPU methods
#   - Average memory usage (GB)
#
# CPU Metrics:
#   - CPU utilization (%)
#   - Number of threads used
#
# GPU Metrics (IDIS, FPS, DEPOCO):
#   - GPU utilization (%)
#   - GPU memory utilization (%)
#   - GPU power consumption (W)
#   - GPU temperature (°C)
#
################################################################################
# GPU Memory Measurement Methodology
################################################################################
#
# GPU memory is measured using nvidia-smi at 0.5-second intervals during
# execution. The reported "Peak GPU Memory" is the maximum observed value
# of `memory.used` during the benchmark run.
#
# Important notes:
#   1. nvidia-smi reports TOTAL device memory usage, not process-specific.
#      However, benchmarks are run on a clean GPU (baseline ~13-17 MB for
#      CUDA driver overhead), so measurements reflect actual method usage.
#
#   2. Memory usage patterns vary by method:
#      - FPS (~0.5 GB): Memory-efficient, processes one scan at a time.
#        Only holds current point cloud + intermediate tensors in GPU memory.
#      - IDIS (~2.2 GB): Moderate usage for distance calculations and
#        density-weighted sampling tensor operations.
#      - DEPOCO (~37 GB): High usage due to neural encoder-decoder architecture.
#        Loads full model weights + batch processing buffers.
#
#   3. For accurate measurements, ensure no other GPU processes are running
#      during benchmarking. Check with: nvidia-smi before starting.
#
#   4. The baseline GPU memory (before method execution) is typically 13-17 MB
#      which is just CUDA driver/context overhead. This is NOT subtracted from
#      the reported peak values.
#
################################################################################

set -euo pipefail

# Trap Ctrl+C
cleanup() {
    echo ""
    echo "Interrupted - cleaning up..."
    # Kill any background monitoring processes
    pkill -f "gpu_monitor_$$" 2>/dev/null || true
    pkill -f "cpu_monitor_$$" 2>/dev/null || true
    exit 130
}
trap cleanup INT TERM

################################################################################
# Configuration
################################################################################

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
SEQUENCES="08"                    # Default: validation sequence only (4,071 scans)
LOSS="70"                         # Default: 70% loss (DEPOCO has verified 70% model)
BENCHMARK_DIR="$PROJECT_ROOT/benchmark_results"
SKIP_EXISTING=false
METHODS="RS,DBSCAN,Voxel,Poisson,IDIS,FPS,DEPOCO"
WORKERS=16                        # Reduced workers for fair comparison
ITERATIONS=3                      # Number of iterations to run

# DEPOCO configuration (uses separate venv)
DEPOCO_VENV="/DATA/aakash/ms-project/venv/py38_depoco"
DEPOCO_BASE="/DATA/aakash/ms-project/depoco_for_transfer"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sequences)
            SEQUENCES="$2"
            shift 2
            ;;
        --loss)
            LOSS="$2"
            shift 2
            ;;
        --output-dir)
            BENCHMARK_DIR="$2"
            shift 2
            ;;
        --skip-existing)
            SKIP_EXISTING=true
            shift
            ;;
        --methods)
            METHODS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -h|--help)
            head -55 "$0" | tail -50
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create benchmark directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$BENCHMARK_DIR/run_$TIMESTAMP"
TEMP_OUTPUT_DIR="$RESULTS_DIR/subsampled_temp"
METRICS_DIR="$RESULTS_DIR/metrics"
LOGS_DIR="$RESULTS_DIR/logs"

mkdir -p "$TEMP_OUTPUT_DIR" "$METRICS_DIR" "$LOGS_DIR"

# Activate environment
echo "Activating environment..."
if [[ -f "$PROJECT_ROOT/ptv3_venv/bin/activate" ]]; then
    source "$PROJECT_ROOT/ptv3_venv/bin/activate"
else
    echo "Error: Virtual environment not found"
    exit 1
fi

cd "$PROJECT_ROOT"

################################################################################
# Helper Functions
################################################################################

# Get current timestamp
timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Log message
log() {
    echo "[$(timestamp)] $1"
}

# Check if nvidia-smi is available
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "true"
    else
        echo "false"
    fi
}

# Start GPU monitoring in background
start_gpu_monitor() {
    local output_file="$1"
    local pid_file="$2"

    # Monitor GPU every 0.5 seconds
    (
        echo "timestamp,gpu_util,mem_util,mem_used_mb,power_w,temp_c" > "$output_file"
        while true; do
            nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu \
                --format=csv,noheader,nounits 2>/dev/null | while read line; do
                echo "$(date +%s.%N),$line" >> "$output_file"
            done
            sleep 0.5
        done
    ) &
    echo $! > "$pid_file"
}

# Stop GPU monitoring
stop_gpu_monitor() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        kill $(cat "$pid_file") 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

# Start CPU/Memory monitoring in background
start_cpu_monitor() {
    local output_file="$1"
    local pid_file="$2"
    local target_pattern="$3"

    (
        echo "timestamp,cpu_percent,mem_rss_mb,mem_vms_mb,threads" > "$output_file"
        while true; do
            # Get stats for python processes matching our pattern
            ps aux | grep -E "$target_pattern" | grep -v grep | awk '{
                cpu+=$3; rss+=$6; vsz+=$5; count++
            } END {
                if (count > 0) {
                    printf "%.3f,%.1f,%.1f,%.1f,%d\n", systime(), cpu, rss/1024, vsz/1024, count
                }
            }' >> "$output_file" 2>/dev/null || true
            sleep 0.5
        done
    ) &
    echo $! > "$pid_file"
}

# Stop CPU monitoring
stop_cpu_monitor() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        kill $(cat "$pid_file") 2>/dev/null || true
        rm -f "$pid_file"
    fi
}

# Format number to ensure leading zero (bc outputs .123 instead of 0.123)
format_number() {
    local num="$1"
    # If number starts with '.', prepend '0'
    if [[ "$num" == .* ]]; then
        echo "0$num"
    else
        echo "$num"
    fi
}

# Parse monitoring data and compute statistics
compute_metrics() {
    local monitor_file="$1"
    local output_file="$2"

    python3 << EOF
import csv
import json
import sys

metrics = {
    'samples': 0,
    'gpu_util_avg': 0, 'gpu_util_max': 0,
    'mem_util_avg': 0, 'mem_util_max': 0,
    'mem_used_mb_avg': 0, 'mem_used_mb_max': 0,
    'power_w_avg': 0, 'power_w_max': 0,
    'temp_c_avg': 0, 'temp_c_max': 0,
    'cpu_percent_avg': 0, 'cpu_percent_max': 0,
    'mem_rss_mb_avg': 0, 'mem_rss_mb_max': 0,
}

try:
    with open('$monitor_file', 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(json.dumps(metrics))
        sys.exit(0)

    metrics['samples'] = len(rows)

    # GPU metrics
    if 'gpu_util' in rows[0]:
        gpu_utils = [float(r['gpu_util']) for r in rows if r['gpu_util'].strip()]
        if gpu_utils:
            metrics['gpu_util_avg'] = sum(gpu_utils) / len(gpu_utils)
            metrics['gpu_util_max'] = max(gpu_utils)

        mem_utils = [float(r['mem_util']) for r in rows if r['mem_util'].strip()]
        if mem_utils:
            metrics['mem_util_avg'] = sum(mem_utils) / len(mem_utils)
            metrics['mem_util_max'] = max(mem_utils)

        mem_used = [float(r['mem_used_mb']) for r in rows if r['mem_used_mb'].strip()]
        if mem_used:
            metrics['mem_used_mb_avg'] = sum(mem_used) / len(mem_used)
            metrics['mem_used_mb_max'] = max(mem_used)

        powers = [float(r['power_w']) for r in rows if r['power_w'].strip()]
        if powers:
            metrics['power_w_avg'] = sum(powers) / len(powers)
            metrics['power_w_max'] = max(powers)

        temps = [float(r['temp_c']) for r in rows if r['temp_c'].strip()]
        if temps:
            metrics['temp_c_avg'] = sum(temps) / len(temps)
            metrics['temp_c_max'] = max(temps)

    # CPU metrics
    if 'cpu_percent' in rows[0]:
        cpu_pcts = [float(r['cpu_percent']) for r in rows if r['cpu_percent'].strip()]
        if cpu_pcts:
            metrics['cpu_percent_avg'] = sum(cpu_pcts) / len(cpu_pcts)
            metrics['cpu_percent_max'] = max(cpu_pcts)

        mem_rss = [float(r['mem_rss_mb']) for r in rows if r['mem_rss_mb'].strip()]
        if mem_rss:
            metrics['mem_rss_mb_avg'] = sum(mem_rss) / len(mem_rss)
            metrics['mem_rss_mb_max'] = max(mem_rss)

except Exception as e:
    print(json.dumps(metrics))
    sys.exit(0)

print(json.dumps(metrics))
EOF
}

# Count scans in sequences
count_scans() {
    local seqs="$1"
    local count=0
    for seq in $seqs; do
        # Try both possible paths
        seq_dir="$PROJECT_ROOT/data/SemanticKITTI/original/sequences/$seq/velodyne"
        if [[ ! -d "$seq_dir" ]]; then
            seq_dir="$PROJECT_ROOT/data/SemanticKITTI/original/dataset/sequences/$seq/velodyne"
        fi
        if [[ -d "$seq_dir" ]]; then
            count=$((count + $(ls -1 "$seq_dir"/*.bin 2>/dev/null | wc -l)))
        fi
    done
    echo $count
}

################################################################################
# Benchmark Functions
################################################################################

# Benchmark CPU method (RS, DBSCAN, Voxel, Poisson)
benchmark_cpu_method() {
    local method="$1"
    local loss="$2"
    local sequences="$3"
    local output_dir="$4"
    local iteration="$5"

    log "Starting benchmark: $method (CPU) - Iteration $iteration"

    local method_output="$output_dir/${method}_loss${loss}"
    mkdir -p "$method_output"

    # Determine if method needs seed
    local seed_arg=""
    if [[ "$method" == "RS" || "$method" == "Poisson" ]]; then
        seed_arg="--seeds 1"
    fi

    # Start CPU monitoring
    local cpu_monitor_file="$METRICS_DIR/${method}_iter${iteration}_cpu_monitor.csv"
    local cpu_pid_file="$METRICS_DIR/${method}_iter${iteration}_cpu.pid"
    start_cpu_monitor "$cpu_monitor_file" "$cpu_pid_file" "generate_subsampled"

    # Run benchmark with time measurement
    local start_time=$(date +%s.%N)

    PYTHONUNBUFFERED=1 /usr/bin/time -v python scripts/preprocessing/generate_subsampled_semantickitti_v2.py \
        --methods $method \
        --loss-levels $loss \
        $seed_arg \
        --workers $WORKERS \
        --sequences $sequences \
        --output-dir "$output_dir" \
        2>&1 | tee "$LOGS_DIR/${method}_iter${iteration}_benchmark.log"

    local end_time=$(date +%s.%N)
    local elapsed=$(format_number "$(echo "$end_time - $start_time" | bc)")

    # Stop monitoring
    stop_cpu_monitor "$cpu_pid_file"

    # Compute metrics
    local cpu_metrics=$(compute_metrics "$cpu_monitor_file" "")

    # Save results for this iteration
    local num_scans=$(count_scans "$sequences")
    if [[ "$num_scans" -eq 0 ]]; then
        log "ERROR: No scans found for sequences: $sequences"
        return 1
    fi
    local time_per_scan=$(format_number "$(echo "scale=6; $elapsed / $num_scans" | bc)")
    local throughput=$(format_number "$(echo "scale=3; $num_scans / $elapsed" | bc)")

    cat > "$METRICS_DIR/${method}_iter${iteration}_results.json" << EOF
{
    "method": "$method",
    "type": "CPU",
    "iteration": $iteration,
    "loss": $loss,
    "sequences": "$sequences",
    "num_scans": $num_scans,
    "workers": $WORKERS,
    "total_time_seconds": $elapsed,
    "time_per_scan_seconds": $time_per_scan,
    "throughput_scans_per_second": $throughput,
    "cpu_metrics": $cpu_metrics
}
EOF

    log "Completed $method iteration $iteration: ${elapsed}s total, ${time_per_scan}s/scan"
}

# Benchmark GPU method (IDIS, FPS)
benchmark_gpu_method() {
    local method="$1"
    local loss="$2"
    local sequences="$3"
    local output_dir="$4"
    local iteration="$5"

    log "Starting benchmark: $method (GPU) - Iteration $iteration"

    local method_output="$output_dir/${method}_loss${loss}"
    mkdir -p "$method_output"

    # Determine if method needs seed
    local seed_arg=""
    if [[ "$method" == "FPS" ]]; then
        seed_arg="--seed 1"
    fi

    # Start GPU monitoring
    local gpu_monitor_file="$METRICS_DIR/${method}_iter${iteration}_gpu_monitor.csv"
    local gpu_pid_file="$METRICS_DIR/${method}_iter${iteration}_gpu.pid"
    if [[ $(check_gpu) == "true" ]]; then
        start_gpu_monitor "$gpu_monitor_file" "$gpu_pid_file"
    fi

    # Start CPU monitoring too
    local cpu_monitor_file="$METRICS_DIR/${method}_iter${iteration}_cpu_monitor.csv"
    local cpu_pid_file="$METRICS_DIR/${method}_iter${iteration}_cpu.pid"
    start_cpu_monitor "$cpu_monitor_file" "$cpu_pid_file" "generate_subsampled"

    # Run benchmark with time measurement
    local start_time=$(date +%s.%N)

    PYTHONUNBUFFERED=1 /usr/bin/time -v python scripts/preprocessing/generate_subsampled_gpu.py \
        --method $method \
        --loss-levels $loss \
        $seed_arg \
        --sequences $sequences \
        --output-dir "$output_dir" \
        2>&1 | tee "$LOGS_DIR/${method}_iter${iteration}_benchmark.log"

    local end_time=$(date +%s.%N)
    local elapsed=$(format_number "$(echo "$end_time - $start_time" | bc)")

    # Stop monitoring
    stop_gpu_monitor "$gpu_pid_file"
    stop_cpu_monitor "$cpu_pid_file"

    # Compute metrics
    local gpu_metrics="{}"
    if [[ -f "$gpu_monitor_file" ]]; then
        gpu_metrics=$(compute_metrics "$gpu_monitor_file" "")
    fi
    local cpu_metrics=$(compute_metrics "$cpu_monitor_file" "")

    # Save results for this iteration
    local num_scans=$(count_scans "$sequences")
    if [[ "$num_scans" -eq 0 ]]; then
        log "ERROR: No scans found for sequences: $sequences"
        return 1
    fi
    local time_per_scan=$(format_number "$(echo "scale=6; $elapsed / $num_scans" | bc)")
    local throughput=$(format_number "$(echo "scale=3; $num_scans / $elapsed" | bc)")

    cat > "$METRICS_DIR/${method}_iter${iteration}_results.json" << EOF
{
    "method": "$method",
    "type": "GPU",
    "iteration": $iteration,
    "loss": $loss,
    "sequences": "$sequences",
    "num_scans": $num_scans,
    "total_time_seconds": $elapsed,
    "time_per_scan_seconds": $time_per_scan,
    "throughput_scans_per_second": $throughput,
    "gpu_metrics": $gpu_metrics,
    "cpu_metrics": $cpu_metrics
}
EOF

    log "Completed $method iteration $iteration: ${elapsed}s total, ${time_per_scan}s/scan"
}

# Benchmark DEPOCO method (uses separate venv)
benchmark_depoco_method() {
    local method="DEPOCO"
    local loss="$1"
    local sequences="$2"
    local output_dir="$3"
    local iteration="$4"

    log "Starting benchmark: $method (GPU - separate venv) - Iteration $iteration"

    # Check DEPOCO availability
    if [[ ! -f "$DEPOCO_VENV/bin/activate" ]]; then
        log "Error: DEPOCO venv not found at $DEPOCO_VENV"
        return 1
    fi

    local method_output="$output_dir/${method}_loss${loss}"
    mkdir -p "$method_output"

    # Start GPU monitoring
    local gpu_monitor_file="$METRICS_DIR/${method}_iter${iteration}_gpu_monitor.csv"
    local gpu_pid_file="$METRICS_DIR/${method}_iter${iteration}_gpu.pid"
    if [[ $(check_gpu) == "true" ]]; then
        start_gpu_monitor "$gpu_monitor_file" "$gpu_pid_file"
    fi

    # Start CPU monitoring too
    local cpu_monitor_file="$METRICS_DIR/${method}_iter${iteration}_cpu_monitor.csv"
    local cpu_pid_file="$METRICS_DIR/${method}_iter${iteration}_cpu.pid"
    start_cpu_monitor "$cpu_monitor_file" "$cpu_pid_file" "generate_subsampled_depoco"

    # Run benchmark with time measurement using DEPOCO venv
    local start_time=$(date +%s.%N)

    # Export DEPOCO environment variables
    export DEPOCO_BASE="$DEPOCO_BASE"
    export DEPOCO_VENV="$DEPOCO_VENV"

    PYTHONUNBUFFERED=1 /usr/bin/time -v "$DEPOCO_VENV/bin/python" \
        scripts/preprocessing/generate_subsampled_depoco.py \
        --loss $loss \
        --sequences $sequences \
        --output-dir "$output_dir" \
        2>&1 | tee "$LOGS_DIR/${method}_iter${iteration}_benchmark.log"

    local end_time=$(date +%s.%N)
    local elapsed=$(format_number "$(echo "$end_time - $start_time" | bc)")

    # Stop monitoring
    stop_gpu_monitor "$gpu_pid_file"
    stop_cpu_monitor "$cpu_pid_file"

    # Compute metrics
    local gpu_metrics="{}"
    if [[ -f "$gpu_monitor_file" ]]; then
        gpu_metrics=$(compute_metrics "$gpu_monitor_file" "")
    fi
    local cpu_metrics=$(compute_metrics "$cpu_monitor_file" "")

    # Save results for this iteration
    local num_scans=$(count_scans "$sequences")
    if [[ "$num_scans" -eq 0 ]]; then
        log "ERROR: No scans found for sequences: $sequences"
        return 1
    fi
    local time_per_scan=$(format_number "$(echo "scale=6; $elapsed / $num_scans" | bc)")
    local throughput=$(format_number "$(echo "scale=3; $num_scans / $elapsed" | bc)")

    cat > "$METRICS_DIR/${method}_iter${iteration}_results.json" << EOF
{
    "method": "$method",
    "type": "GPU",
    "iteration": $iteration,
    "loss": $loss,
    "sequences": "$sequences",
    "num_scans": $num_scans,
    "total_time_seconds": $elapsed,
    "time_per_scan_seconds": $time_per_scan,
    "throughput_scans_per_second": $throughput,
    "gpu_metrics": $gpu_metrics,
    "cpu_metrics": $cpu_metrics
}
EOF

    log "Completed $method iteration $iteration: ${elapsed}s total, ${time_per_scan}s/scan"
}

################################################################################
# Generate Summary Table with Averages
################################################################################

generate_summary() {
    log "Generating summary table with averaged results..."

    python3 << 'EOF'
import json
import os
import glob
import re
from collections import defaultdict

metrics_dir = os.environ.get('METRICS_DIR', 'benchmark_results/metrics')
results_dir = os.environ.get('RESULTS_DIR', 'benchmark_results')
iterations = int(os.environ.get('ITERATIONS', '3'))

# Collect all iteration results grouped by method
method_results = defaultdict(list)
for json_file in sorted(glob.glob(f"{metrics_dir}/*_iter*_results.json")):
    with open(json_file, 'r') as f:
        content = f.read()
        # Fix invalid JSON numbers like .057730 -> 0.057730 (bc output without leading zero)
        content = re.sub(r':\s*\.(\d)', r': 0.\1', content)
        data = json.loads(content)
        method_results[data['method']].append(data)

if not method_results:
    print("No benchmark results found")
    exit(0)

# Compute averages for each method
averaged_results = []
all_iterations = []  # Store all individual iterations too

for method, runs in method_results.items():
    if not runs:
        continue

    # Store individual runs
    for run in runs:
        all_iterations.append(run)

    n = len(runs)

    # Average timing metrics
    avg_total_time = sum(r['total_time_seconds'] for r in runs) / n
    avg_time_per_scan = sum(r['time_per_scan_seconds'] for r in runs) / n
    avg_throughput = sum(r['throughput_scans_per_second'] for r in runs) / n

    # Standard deviation for timing
    if n > 1:
        std_total_time = (sum((r['total_time_seconds'] - avg_total_time)**2 for r in runs) / (n-1)) ** 0.5
        std_time_per_scan = (sum((r['time_per_scan_seconds'] - avg_time_per_scan)**2 for r in runs) / (n-1)) ** 0.5
    else:
        std_total_time = 0
        std_time_per_scan = 0

    # Average CPU metrics
    cpu_metrics_avg = {}
    cpu_keys = ['cpu_percent_avg', 'cpu_percent_max', 'mem_rss_mb_avg', 'mem_rss_mb_max']
    for key in cpu_keys:
        values = [r.get('cpu_metrics', {}).get(key, 0) for r in runs]
        cpu_metrics_avg[key] = sum(values) / n if values else 0

    # Average GPU metrics (for GPU methods)
    gpu_metrics_avg = {}
    if runs[0]['type'] == 'GPU':
        gpu_keys = ['gpu_util_avg', 'gpu_util_max', 'mem_util_avg', 'mem_util_max',
                    'mem_used_mb_avg', 'mem_used_mb_max', 'power_w_avg', 'power_w_max',
                    'temp_c_avg', 'temp_c_max']
        for key in gpu_keys:
            values = [r.get('gpu_metrics', {}).get(key, 0) for r in runs]
            gpu_metrics_avg[key] = sum(values) / n if values else 0

    avg_result = {
        'method': method,
        'type': runs[0]['type'],
        'loss': runs[0]['loss'],
        'sequences': runs[0]['sequences'],
        'num_scans': runs[0]['num_scans'],
        'iterations': n,
        'total_time_seconds': avg_total_time,
        'total_time_std': std_total_time,
        'time_per_scan_seconds': avg_time_per_scan,
        'time_per_scan_std': std_time_per_scan,
        'throughput_scans_per_second': avg_throughput,
        'cpu_metrics': cpu_metrics_avg,
        'gpu_metrics': gpu_metrics_avg,
        'individual_runs': [
            {
                'iteration': r.get('iteration', i+1),
                'total_time_seconds': r['total_time_seconds'],
                'time_per_scan_seconds': r['time_per_scan_seconds'],
                'throughput_scans_per_second': r['throughput_scans_per_second']
            }
            for i, r in enumerate(runs)
        ]
    }

    if 'workers' in runs[0]:
        avg_result['workers'] = runs[0]['workers']

    averaged_results.append(avg_result)

# Sort by method type and time
averaged_results.sort(key=lambda x: (x['type'], x['total_time_seconds']))

# Generate summary table
print("\n" + "="*120)
print("SUBSAMPLING EFFICIENCY BENCHMARK RESULTS (AVERAGED OVER {} ITERATIONS)".format(iterations))
print("="*120)

# Header
print(f"\n{'Method':<10} {'Type':<5} {'Iters':<6} {'Total(s)':<12} {'Std':<8} {'Per Scan(s)':<12} {'Throughput':<12} {'Peak RAM(GB)':<12} {'Peak GPU(GB)':<12}")
print("-"*120)

for r in averaged_results:
    method = r['method']
    mtype = r['type']
    iters = r['iterations']
    total = r['total_time_seconds']
    total_std = r['total_time_std']
    per_scan = r['time_per_scan_seconds']
    throughput = r['throughput_scans_per_second']

    # Get memory metrics
    cpu_metrics = r.get('cpu_metrics', {})
    gpu_metrics = r.get('gpu_metrics', {})

    peak_ram = cpu_metrics.get('mem_rss_mb_max', 0) / 1024  # Convert to GB
    peak_gpu = gpu_metrics.get('mem_used_mb_max', 0) / 1024  # Convert to GB

    print(f"{method:<10} {mtype:<5} {iters:<6} {total:<12.2f} {total_std:<8.2f} {per_scan:<12.4f} {throughput:<12.2f} {peak_ram:<12.2f} {peak_gpu:<12.2f}")

print("-"*120)

# Show individual iteration times
print("\nIndividual Iteration Times:")
print("-"*80)
for r in averaged_results:
    method = r['method']
    runs = r['individual_runs']
    times = [f"{run['total_time_seconds']:.2f}s" for run in runs]
    print(f"  {method:<10}: {', '.join(times)}")

# Additional GPU metrics if available
gpu_methods = [r for r in averaged_results if r['type'] == 'GPU']
if gpu_methods:
    print("\nGPU Metrics (Averaged):")
    print(f"{'Method':<10} {'GPU Util(%)':<12} {'Mem Util(%)':<12} {'Power(W)':<10} {'Temp(C)':<10}")
    print("-"*60)
    for r in gpu_methods:
        gm = r.get('gpu_metrics', {})
        print(f"{r['method']:<10} {gm.get('gpu_util_avg', 0):<12.1f} {gm.get('mem_util_avg', 0):<12.1f} {gm.get('power_w_avg', 0):<10.1f} {gm.get('temp_c_avg', 0):<10.1f}")

# Save averaged results as CSV
csv_file = f"{results_dir}/benchmark_summary.csv"
with open(csv_file, 'w') as f:
    f.write("method,type,iterations,total_time_s,total_time_std,time_per_scan_s,throughput_scans_s,peak_ram_gb,peak_gpu_gb,gpu_util_avg,gpu_mem_util_avg,gpu_power_w_avg\n")
    for r in averaged_results:
        cpu_metrics = r.get('cpu_metrics', {})
        gpu_metrics = r.get('gpu_metrics', {})
        f.write(f"{r['method']},{r['type']},{r['iterations']},{r['total_time_seconds']:.2f},{r['total_time_std']:.2f},{r['time_per_scan_seconds']:.4f},{r['throughput_scans_per_second']:.2f},")
        f.write(f"{cpu_metrics.get('mem_rss_mb_max', 0)/1024:.2f},{gpu_metrics.get('mem_used_mb_max', 0)/1024:.2f},")
        f.write(f"{gpu_metrics.get('gpu_util_avg', 0):.1f},{gpu_metrics.get('mem_util_avg', 0):.1f},{gpu_metrics.get('power_w_avg', 0):.1f}\n")

print(f"\nSummary saved to: {csv_file}")

# Save detailed results as JSON (includes both averaged and individual)
json_file = f"{results_dir}/benchmark_summary.json"
with open(json_file, 'w') as f:
    json.dump({
        'averaged_results': averaged_results,
        'all_iterations': all_iterations
    }, f, indent=2)
print(f"Detailed results saved to: {json_file}")

print("\n" + "="*120)
EOF
}

################################################################################
# Main Execution
################################################################################

echo ""
echo "================================================================"
echo "  Subsampling Efficiency Benchmark"
echo "================================================================"
echo ""
log "Started"
echo ""
echo "Configuration:"
echo "  Sequences:    $SEQUENCES"
echo "  Loss level:   $LOSS%"
echo "  Methods:      $METHODS"
echo "  Workers:      $WORKERS (for CPU methods)"
echo "  Iterations:   $ITERATIONS"
echo "  Output dir:   $RESULTS_DIR"
echo ""

# Count total scans
TOTAL_SCANS=$(count_scans "$SEQUENCES")
echo "Total scans to process: $TOTAL_SCANS"
echo ""

# Check GPU availability
HAS_GPU=$(check_gpu)
if [[ "$HAS_GPU" == "true" ]]; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
else
    echo "No GPU detected - skipping GPU methods"
fi
echo ""

# Export variables for Python scripts
export METRICS_DIR
export RESULTS_DIR
export ITERATIONS

# Run benchmarks for each method, multiple iterations
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"

for iteration in $(seq 1 $ITERATIONS); do
    echo ""
    echo "================================================================"
    echo "  ITERATION $iteration of $ITERATIONS"
    echo "================================================================"

    # Clear temp directory at the start of each iteration
    log "Clearing temp directory for iteration $iteration..."
    rm -rf "$TEMP_OUTPUT_DIR"/*
    log "Temp directory cleared"

    for method in "${METHOD_ARRAY[@]}"; do
        method=$(echo "$method" | xargs)  # Trim whitespace

        echo ""
        echo "----------------------------------------------------------------"
        echo "  Benchmarking: $method (Iteration $iteration)"
        echo "----------------------------------------------------------------"

        # Clear method-specific output directory before each benchmark
        local_method_dir="$TEMP_OUTPUT_DIR/${method}_loss${LOSS}"
        if [[ -d "$local_method_dir" ]]; then
            log "Clearing existing data for $method..."
            rm -rf "$local_method_dir"
        fi

        case $method in
            RS|DBSCAN|Voxel|Poisson)
                benchmark_cpu_method "$method" "$LOSS" "$SEQUENCES" "$TEMP_OUTPUT_DIR" "$iteration"
                ;;
            IDIS|FPS)
                if [[ "$HAS_GPU" == "true" ]]; then
                    benchmark_gpu_method "$method" "$LOSS" "$SEQUENCES" "$TEMP_OUTPUT_DIR" "$iteration"
                else
                    log "Skipping $method - no GPU available"
                fi
                ;;
            DEPOCO)
                if [[ "$HAS_GPU" == "true" ]]; then
                    benchmark_depoco_method "$LOSS" "$SEQUENCES" "$TEMP_OUTPUT_DIR" "$iteration"
                else
                    log "Skipping $method - no GPU available"
                fi
                ;;
            *)
                log "Unknown method: $method"
                ;;
        esac
    done
done

# Generate summary with averages
echo ""
echo "----------------------------------------------------------------"
echo "  Generating Summary (Averaged over $ITERATIONS iterations)"
echo "----------------------------------------------------------------"
generate_summary

# Cleanup temp subsampled data (optional - comment out to keep)
# rm -rf "$TEMP_OUTPUT_DIR"

echo ""
echo "================================================================"
echo "  Benchmark Complete"
echo "================================================================"
echo ""
log "Finished"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "  - benchmark_summary.csv  (averaged results for plotting)"
echo "  - benchmark_summary.json (detailed metrics with all iterations)"
echo "  - metrics/*_iter*_results.json (per-iteration results)"
echo "  - logs/*_iter*_benchmark.log   (execution logs)"
echo ""
