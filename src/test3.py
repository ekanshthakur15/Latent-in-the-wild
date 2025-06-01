# test.py
import os
import re
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
import argparse
import logging
from pathlib import Path
import json
from datetime import datetime

from fingerprint_pipeline import FingerprintMatchingPipeline


class FingerprintTester:
    """Test fingerprint matching pipeline and calculate performance metrics"""

    def __init__(self, dataset_path, max_subjects=20, pairs_per_type=500):
        """
        Initialize tester

        Args:
            dataset_path: Path to Images directory
            max_subjects: Maximum number of subjects to use
            pairs_per_type: Number of genuine/impostor pairs to generate
        """
        self.dataset_path = Path(dataset_path)
        self.max_subjects = max_subjects
        self.pairs_per_type = pairs_per_type

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Initialize pipeline
        self.pipeline = FingerprintMatchingPipeline()

        # Results storage
        self.genuine_scores = []
        self.impostor_scores = []
        self.results = {}

    def parse_filename(self, filename):
        """
        Parse filename to extract subject, finger, and surface info

        Expected format: Sub015_Left-Hand_IndexFinger_Metal_Cropped
        Flexible pattern: Sub{XXX}_{Left|Right}-Hand_{Finger}Finger_{Surface}_{Optional}
        """
        try:
            # Main pattern for the expected filename format
            pattern = r"Sub(\d+)_(Left|Right)-Hand_(\w+)Finger_(\w+)(?:_\w+)*"
            match = re.search(pattern, filename, re.IGNORECASE)

            if match:
                subject_id = match.group(1).zfill(3)  # Pad with zeros if needed
                hand = match.group(2).lower()
                finger = match.group(3).lower()
                surface = match.group(4).lower()

                finger_id = f"{hand}_{finger}"
                full_finger_id = f"Sub{subject_id}_{finger_id}"

                return {
                    "subject_id": subject_id,
                    "hand": hand,
                    "finger": finger,
                    "finger_id": finger_id,
                    "surface": surface,
                    "full_finger_id": full_finger_id,
                }

            # Fallback pattern for variations
            fallback_pattern = r"Sub(\d+).*?(Left|Right).*?(\w+)Finger"
            fallback_match = re.search(fallback_pattern, filename, re.IGNORECASE)

            if fallback_match:
                subject_id = fallback_match.group(1).zfill(3)
                hand = fallback_match.group(2).lower()
                finger = fallback_match.group(3).lower()

                # Try to extract surface information
                surface = "unknown"
                surface_patterns = [
                    "wall",
                    "ipad",
                    "mobile",
                    "glass",
                    "paper",
                    "screen",
                    "metal",
                    "wood",
                    "plastic",
                ]
                for surf in surface_patterns:
                    if surf in filename.lower():
                        surface = surf
                        break

                finger_id = f"{hand}_{finger}"
                full_finger_id = f"Sub{subject_id}_{finger_id}"

                self.logger.warning(f"Used fallback parsing for: {filename}")

                return {
                    "subject_id": subject_id,
                    "hand": hand,
                    "finger": finger,
                    "finger_id": finger_id,
                    "surface": surface,
                    "full_finger_id": full_finger_id,
                }

            return None

        except Exception as e:
            self.logger.warning(f"Could not parse filename {filename}: {e}")
            return None

    def find_all_images(self, root_path):
        """
        Recursively find all image files in the directory structure

        Args:
            root_path: Root directory to search

        Returns:
            List of Path objects for all image files
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = []

        for file_path in root_path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        return image_files

    def load_dataset(self):
        """Load and organize dataset with flexible folder structure"""
        self.logger.info(f"Loading dataset from {self.dataset_path}")

        # Dictionary to store images by subject and finger
        dataset = defaultdict(lambda: defaultdict(list))

        # Get all subject directories (Sub001, Sub002, etc.)
        subject_dirs = []

        # First, try to find subject directories at the root level
        for item in self.dataset_path.iterdir():
            if item.is_dir() and re.match(r"Sub\d+", item.name, re.IGNORECASE):
                subject_dirs.append(item)

        # If no subject directories found at root, search recursively
        if not subject_dirs:
            for item in self.dataset_path.rglob("*"):
                if item.is_dir() and re.match(r"Sub\d+", item.name, re.IGNORECASE):
                    subject_dirs.append(item)

        # Limit number of subjects
        if len(subject_dirs) > self.max_subjects:
            subject_dirs = sorted(subject_dirs, key=lambda x: x.name)[
                : self.max_subjects
            ]

        self.logger.info(f"Found {len(subject_dirs)} subject directories")

        total_processed = 0
        total_valid = 0

        # Process each subject directory
        for subject_dir in subject_dirs:
            subject_name = subject_dir.name
            self.logger.info(f"Processing subject: {subject_name}")

            # Find all image files recursively within this subject directory
            image_files = self.find_all_images(subject_dir)

            self.logger.info(
                f"  Found {len(image_files)} image files in {subject_name}"
            )

            # Process each image file
            for image_file in image_files:
                total_processed += 1

                # Parse filename to extract fingerprint information
                parsed = self.parse_filename(image_file.name)

                if parsed:
                    total_valid += 1

                    # Use the full finger ID as the key
                    finger_key = parsed["full_finger_id"]

                    # Store image information
                    if finger_key not in dataset:
                        dataset[finger_key] = {
                            "images": [],
                            "subject_id": parsed["subject_id"],
                            "hand": parsed["hand"],
                            "finger": parsed["finger"],
                            "finger_id": parsed["finger_id"],
                            "surfaces": set(),
                        }

                    dataset[finger_key]["images"].append(str(image_file))
                    dataset[finger_key]["surfaces"].add(parsed["surface"])
                else:
                    self.logger.debug(
                        f"  Skipped file (parsing failed): {image_file.name}"
                    )

        # Convert sets to lists for JSON serialization later
        for finger_data in dataset.values():
            if "surfaces" in finger_data:
                finger_data["surfaces"] = list(finger_data["surfaces"])

        # Filter out fingers with less than 2 images (can't create pairs)
        filtered_dataset = {k: v for k, v in dataset.items() if len(v["images"]) >= 2}

        # Log statistics
        total_images = sum(len(v["images"]) for v in filtered_dataset.values())
        subjects_found = len(set(v["subject_id"] for v in filtered_dataset.values()))

        self.logger.info(f"\nDataset Statistics:")
        self.logger.info(f"  Total files processed: {total_processed}")
        self.logger.info(f"  Valid fingerprint images: {total_valid}")
        self.logger.info(f"  Unique subjects: {subjects_found}")
        self.logger.info(f"  Unique fingers (with ≥2 images): {len(filtered_dataset)}")
        self.logger.info(f"  Total usable images: {total_images}")

        # Log finger distribution by subject
        subject_finger_count = defaultdict(int)
        for finger_data in filtered_dataset.values():
            subject_finger_count[finger_data["subject_id"]] += 1

        self.logger.info(f"\nFingers per subject:")
        for subject_id in sorted(subject_finger_count.keys()):
            count = subject_finger_count[subject_id]
            self.logger.info(f"  Sub{subject_id}: {count} fingers")

        # Log surface distribution
        surface_count = defaultdict(int)
        hand_count = defaultdict(int)
        finger_type_count = defaultdict(int)

        for finger_data in filtered_dataset.values():
            for surface in finger_data.get("surfaces", []):
                surface_count[surface] += 1
            hand_count[finger_data["hand"]] += 1
            finger_type_count[finger_data["finger"]] += 1

        self.logger.info(f"\nSurface distribution:")
        for surface, count in sorted(surface_count.items()):
            self.logger.info(f"  {surface}: {count} fingers")

        self.logger.info(f"\nHand distribution:")
        for hand, count in sorted(hand_count.items()):
            self.logger.info(f"  {hand}: {count} fingers")

        self.logger.info(f"\nFinger type distribution:")
        for finger_type, count in sorted(finger_type_count.items()):
            self.logger.info(f"  {finger_type}: {count} fingers")

        if not filtered_dataset:
            raise ValueError(
                "No valid fingerprint images found with the expected naming pattern"
            )

        return filtered_dataset

    def generate_genuine_pairs(self, dataset):
        """Generate genuine pairs (same finger, different images)"""
        genuine_pairs = []

        for finger_id, finger_data in dataset.items():
            images = finger_data["images"]
            if len(images) >= 2:
                # Generate all possible pairs for this finger
                for img1, img2 in combinations(images, 2):
                    genuine_pairs.append((img1, img2, "genuine"))

        # Randomly sample if we have too many pairs
        if len(genuine_pairs) > self.pairs_per_type:
            genuine_pairs = random.sample(genuine_pairs, self.pairs_per_type)

        self.logger.info(f"Generated {len(genuine_pairs)} genuine pairs")
        return genuine_pairs

    def generate_impostor_pairs(self, dataset):
        """Generate impostor pairs (different subjects)"""
        impostor_pairs = []

        # Group images by subject
        subjects = defaultdict(list)
        for finger_id, finger_data in dataset.items():
            subjects[finger_data["subject_id"]].extend(finger_data["images"])

        # Generate pairs between different subjects
        subject_ids = list(subjects.keys())

        if len(subject_ids) < 2:
            raise ValueError("Need at least 2 subjects to generate impostor pairs")

        for _ in range(self.pairs_per_type):
            # Pick two different subjects
            subj1, subj2 = random.sample(subject_ids, 2)

            # Pick random images from each subject
            img1 = random.choice(subjects[subj1])
            img2 = random.choice(subjects[subj2])

            impostor_pairs.append((img1, img2, "impostor"))

        self.logger.info(f"Generated {len(impostor_pairs)} impostor pairs")
        return impostor_pairs

    def run_comparisons(self, pairs):
        """Run fingerprint comparisons for given pairs"""
        results = []

        total_pairs = len(pairs)
        for i, (img1, img2, pair_type) in enumerate(pairs):
            try:
                if (i + 1) % 50 == 0:
                    self.logger.info(
                        f"Processing pair {i+1}/{total_pairs} ({pair_type})"
                    )

                # Run comparison
                result = self.pipeline.match_fingerprints(img1, img2)

                score_data = {
                    "image1": img1,
                    "image2": img2,
                    "type": pair_type,
                    "overall_score": result["overall_score"],
                    "minutiae_score": result["minutiae_score"],
                    "orb_score": result["orb_score"],
                    "is_match": result["is_match"],
                }

                results.append(score_data)

                # Store scores by type
                if pair_type == "genuine":
                    self.genuine_scores.append(result["overall_score"])
                else:
                    self.impostor_scores.append(result["overall_score"])

            except Exception as e:
                self.logger.error(f"Error processing pair {img1} vs {img2}: {e}")
                continue

        return results

    def calculate_eer(self):
        """Calculate Equal Error Rate (EER)"""
        if not self.genuine_scores or not self.impostor_scores:
            return None, None, None

        # Create arrays
        genuine_scores = np.array(self.genuine_scores)
        impostor_scores = np.array(self.impostor_scores)

        # Find threshold range
        min_score = min(np.min(genuine_scores), np.min(impostor_scores))
        max_score = max(np.max(genuine_scores), np.max(impostor_scores))

        thresholds = np.linspace(min_score, max_score, 1000)

        far_rates = []  # False Accept Rate
        frr_rates = []  # False Reject Rate

        for threshold in thresholds:
            # False Accept Rate: impostor scores >= threshold
            far = np.sum(impostor_scores >= threshold) / len(impostor_scores)

            # False Reject Rate: genuine scores < threshold
            frr = np.sum(genuine_scores < threshold) / len(genuine_scores)

            far_rates.append(far)
            frr_rates.append(frr)

        far_rates = np.array(far_rates)
        frr_rates = np.array(frr_rates)

        # Find EER (where FAR ≈ FRR)
        eer_idx = np.argmin(np.abs(far_rates - frr_rates))
        eer = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]

        return eer, eer_threshold, (thresholds, far_rates, frr_rates)

    def plot_results(self, save_path=None):
        """Plot score distributions and ROC curve"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Score distributions
        ax1.hist(
            self.genuine_scores,
            bins=50,
            alpha=0.7,
            label="Genuine",
            color="green",
            density=True,
        )
        ax1.hist(
            self.impostor_scores,
            bins=50,
            alpha=0.7,
            label="Impostor",
            color="red",
            density=True,
        )
        ax1.set_xlabel("Score")
        ax1.set_ylabel("Density")
        ax1.set_title("Score Distributions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plots
        ax2.boxplot(
            [self.genuine_scores, self.impostor_scores], labels=["Genuine", "Impostor"]
        )
        ax2.set_ylabel("Score")
        ax2.set_title("Score Box Plots")
        ax2.grid(True, alpha=0.3)

        # ROC Curve
        if hasattr(self, "roc_data") and self.roc_data:
            thresholds, far_rates, frr_rates = self.roc_data
            ax3.plot(far_rates, 1 - frr_rates, "b-", linewidth=2)
            ax3.plot([0, 1], [0, 1], "r--", alpha=0.5)
            ax3.set_xlabel("False Accept Rate (FAR)")
            ax3.set_ylabel("True Accept Rate (1-FRR)")
            ax3.set_title("ROC Curve")
            ax3.grid(True, alpha=0.3)

        # FAR/FRR vs Threshold
        if hasattr(self, "roc_data") and self.roc_data:
            thresholds, far_rates, frr_rates = self.roc_data
            ax4.plot(thresholds, far_rates, "r-", label="FAR", linewidth=2)
            ax4.plot(thresholds, frr_rates, "b-", label="FRR", linewidth=2)
            if hasattr(self, "eer_threshold"):
                ax4.axvline(
                    x=self.eer_threshold,
                    color="g",
                    linestyle="--",
                    label=f"EER Threshold ({self.eer_threshold:.3f})",
                )
            ax4.set_xlabel("Threshold")
            ax4.set_ylabel("Error Rate")
            ax4.set_title("FAR/FRR vs Threshold")
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Plots saved to {save_path}")

        plt.show()

    def save_results(self, output_file):
        """Save detailed results to JSON file"""
        results_data = {
            "test_info": {
                "dataset_path": str(self.dataset_path),
                "max_subjects": self.max_subjects,
                "pairs_per_type": self.pairs_per_type,
                "timestamp": datetime.now().isoformat(),
            },
            "statistics": {
                "genuine_pairs": len(self.genuine_scores),
                "impostor_pairs": len(self.impostor_scores),
                "genuine_mean": (
                    float(np.mean(self.genuine_scores)) if self.genuine_scores else 0
                ),
                "genuine_std": (
                    float(np.std(self.genuine_scores)) if self.genuine_scores else 0
                ),
                "impostor_mean": (
                    float(np.mean(self.impostor_scores)) if self.impostor_scores else 0
                ),
                "impostor_std": (
                    float(np.std(self.impostor_scores)) if self.impostor_scores else 0
                ),
                "eer": float(self.eer) if hasattr(self, "eer") and self.eer else None,
                "eer_threshold": (
                    float(self.eer_threshold)
                    if hasattr(self, "eer_threshold") and self.eer_threshold
                    else None
                ),
            },
            "scores": {
                "genuine_scores": [float(s) for s in self.genuine_scores],
                "impostor_scores": [float(s) for s in self.impostor_scores],
            },
        }

        with open(output_file, "w") as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Results saved to {output_file}")

    def run_test(self, output_dir=None):
        """Run complete test suite"""
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

        self.logger.info("Starting fingerprint matching evaluation")

        # Load dataset
        dataset = self.load_dataset()
        if not dataset:
            raise ValueError("No valid fingerprint images found")

        # Generate pairs
        genuine_pairs = self.generate_genuine_pairs(dataset)
        impostor_pairs = self.generate_impostor_pairs(dataset)

        if not genuine_pairs or not impostor_pairs:
            raise ValueError("Could not generate sufficient test pairs")

        # Run comparisons
        self.logger.info("Running genuine comparisons...")
        genuine_results = self.run_comparisons(genuine_pairs)

        self.logger.info("Running impostor comparisons...")
        impostor_results = self.run_comparisons(impostor_pairs)

        # Calculate metrics
        self.eer, self.eer_threshold, self.roc_data = self.calculate_eer()

        # Print results
        self.print_summary()

        # Save results if output directory specified
        if output_dir:
            # Save detailed results
            self.save_results(output_dir / "test_results.json")

            # Save plots
            self.plot_results(output_dir / "performance_plots.png")
        else:
            # Just show plots
            self.plot_results()

        return {
            "eer": self.eer,
            "eer_threshold": self.eer_threshold,
            "genuine_scores": self.genuine_scores,
            "impostor_scores": self.impostor_scores,
            "genuine_mean": np.mean(self.genuine_scores),
            "impostor_mean": np.mean(self.impostor_scores),
        }

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("FINGERPRINT MATCHING EVALUATION RESULTS")
        print("=" * 60)
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Subjects Used: {self.max_subjects}")
        print(f"Genuine Pairs: {len(self.genuine_scores)}")
        print(f"Impostor Pairs: {len(self.impostor_scores)}")
        print()

        if self.genuine_scores:
            print(
                f"Genuine Scores - Mean: {np.mean(self.genuine_scores):.4f}, Std: {np.std(self.genuine_scores):.4f}"
            )
            print(
                f"Genuine Scores - Min: {np.min(self.genuine_scores):.4f}, Max: {np.max(self.genuine_scores):.4f}"
            )

        if self.impostor_scores:
            print(
                f"Impostor Scores - Mean: {np.mean(self.impostor_scores):.4f}, Std: {np.std(self.impostor_scores):.4f}"
            )
            print(
                f"Impostor Scores - Min: {np.min(self.impostor_scores):.4f}, Max: {np.max(self.impostor_scores):.4f}"
            )

        if self.eer is not None:
            print(f"\nEqual Error Rate (EER): {self.eer:.4f} ({self.eer*100:.2f}%)")
            print(f"EER Threshold: {self.eer_threshold:.4f}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test fingerprint matching pipeline")
    parser.add_argument("dataset_path", help="Path to Images directory")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=20,
        help="Maximum number of subjects to use (default: 20)",
    )
    parser.add_argument(
        "--pairs-per-type",
        type=int,
        default=500,
        help="Number of genuine/impostor pairs (default: 500)",
    )
    parser.add_argument("--output-dir", help="Directory to save results")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        # Initialize tester
        tester = FingerprintTester(
            dataset_path=args.dataset_path,
            max_subjects=args.max_subjects,
            pairs_per_type=args.pairs_per_type,
        )

        # Run test
        results = tester.run_test(args.output_dir)

        print(f"\nTest completed successfully!")
        if results["eer"] is not None:
            print(f"Final EER: {results['eer']:.4f} ({results['eer']*100:.2f}%)")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
