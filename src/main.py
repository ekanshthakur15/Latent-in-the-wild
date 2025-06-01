import argparse
import sys
import os
from fingerprint_pipeline import FingerprintMatchingPipeline


def main():
    parser = argparse.ArgumentParser(description="Fingerprint Matching Pipeline")
    parser.add_argument("image1", help="Path to first fingerprint image")
    parser.add_argument("image2", help="Path to second fingerprint image")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Matching threshold (default: 0.6)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Validate input files
    if not os.path.exists(args.image1):
        print(f"Error: First image not found: {args.image1}")
        sys.exit(1)

    if not os.path.exists(args.image2):
        print(f"Error: Second image not found: {args.image2}")
        sys.exit(1)

    try:
        # Initialize pipeline
        pipeline = FingerprintMatchingPipeline()
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

        # Exit with appropriate code
        sys.exit(0 if result["is_match"] else 1)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
