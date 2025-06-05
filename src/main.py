import argparse
import sys
import os
import torch
from fingerprint_pipeline import FingerprintMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Matching Pipeline")
    parser.add_argument("image1", help="Path to first fingerprint image")
    parser.add_argument("image2", help="Path to second fingerprint image")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Matching threshold (default: 0.6)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--use-cuda", action="store_true", help="Use CUDA (GPU) if available"
    )

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.image1):
        print(f"Error: First image not found: {args.image1}")
        sys.exit(1)

    if not os.path.exists(args.image2):
        print(f"Error: Second image not found: {args.image2}")
        sys.exit(1)

    # Determine log level
    log_level = 10 if args.verbose else 20  # logging.DEBUG or logging.INFO

    # Determine device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] CUDA is available. Using GPU.")
    elif args.use_cuda:
        device = torch.device("cpu")
        print("[WARNING] CUDA requested but not available. Using CPU instead.")
    else:
        device = torch.device("cpu")
        print("[INFO] Using CPU.")

    try:
        # Initialize pipeline with device and log level
        pipeline = FingerprintMatchingPipeline(log_level=log_level)
        pipeline.device = device  # if internal classes rely on pipeline.device
        pipeline.matcher.threshold = args.threshold

        # Run matching
        result = pipeline.match_fingerprints(args.image1, args.image2)

        # Display results
        print("\n" + "=" * 50)
        print("FINGERPRINT MATCHING RESULTS")
        print("=" * 50)
        print(f"Image 1: {args.image1}")
        print(f"Image 2: {args.image2}")
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Match Result: {'MATCH' if result['is_match'] else 'NO MATCH'}")
        print(f"Minutiae Score: {result['minutiae_score']:.3f}")
        print(f"ORB Features Score: {result['orb_score']:.3f}")
        print(f"Minutiae Matches: {result['minutiae_matches']}")
        print(f"ORB Matches: {result['orb_matches']}")
        print(f"Threshold: {args.threshold}")
        print("=" * 50)

        sys.exit(0 if result["is_match"] else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()