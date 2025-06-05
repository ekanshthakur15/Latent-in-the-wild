import argparse
import sys
import csv
from pathlib import Path
import torch
import multiprocessing
from fingerprint_pipeline import FingerprintMatchingPipeline
from datetime import datetime
from itertools import combinations

# Global pipeline instance for each worker
pipeline = None

def init_worker(log_level, threshold, device_str):
    global pipeline
    device = torch.device(device_str)
    pipeline = FingerprintMatchingPipeline(log_level=log_level)
    pipeline.device = device
    pipeline.matcher.threshold = threshold

def match_pair(pair):
    global pipeline
    img1, img2 = pair
    try:
        result = pipeline.match_fingerprints(img1, img2)
        return (img1, img2, result["overall_score"])
    except Exception as e:
        print(f"[ERROR] Matching {img1} vs {img2}: {e}")
        return (img1, img2, "ERROR")

def get_image_files(folder_path, extensions=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    return sorted([str(p) for p in Path(folder_path).rglob("*") if p.suffix.lower() in extensions])

def main():
    start_time = datetime.now()

    parser = argparse.ArgumentParser(description="Parallel Fingerprint Matching with Real-Time CSV Writing")
    parser.add_argument("folder", help="Path to folder containing fingerprint images")
    parser.add_argument("--threshold", type=float, default=0.6, help="Matching threshold")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA (GPU) if available")
    parser.add_argument("--output-csv", default="comparison_scores.csv", help="Path to output CSV")
    parser.add_argument("--num-workers", type=int, default=multiprocessing.cpu_count(), help="Number of worker processes (default: max available)")

    args = parser.parse_args()

    log_level = 10 if args.verbose else 20
    device_str = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device_str.upper()}")
    print(f"[INFO] Number of workers: {args.num_workers}")

    image_paths = get_image_files(args.folder)
    if len(image_paths) < 2:
        print("[ERROR] Need at least 2 images for comparison.")
        sys.exit(1)

    pairs = list(combinations(image_paths, 2))
    print(f"[INFO] {len(pairs)} image pairs to compare.")

    with open(args.output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image1_path", "image2_path", "comparison_score"])

        with multiprocessing.Pool(
            processes=args.num_workers,
            initializer=init_worker,
            initargs=(log_level, args.threshold, device_str)
        ) as pool:
            for result in pool.imap_unordered(match_pair, pairs):
                writer.writerow(result)
                csvfile.flush()  # Write immediately
                print(f"[PROCESSED] {result[0]} vs {result[1]} -> Score: {result[2]}")

    total_time = datetime.now() - start_time
    print(f"[INFO] Completed. Results written to {args.output_csv}")
    print(f"[INFO] {len(pairs)} image pairs are compared.")
    print(f"Total time taken to run is: {total_time}")
    print(f"worker count: {args.num_workers}")

if __name__ == "__main__":
    main()