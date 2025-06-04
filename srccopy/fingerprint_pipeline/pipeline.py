import os
import logging
import torch  # For CUDA detection
from .preprocessor import FingerprintPreprocessor
from .feature_extractor import FingerprintFeatureExtractor
from .matcher import FingerprintMatcher


class FingerprintMatchingPipeline:
    """End-to-end fingerprint matching pipeline with optional CUDA acceleration"""

    def __init__(self, log_level=logging.INFO):
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger(__name__)

        # Detect if CUDA is available
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
            self.logger.info("CUDA is available. Using GPU acceleration.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("CUDA not available. Using CPU.")

        # Initialize components safely
        try:
            self.preprocessor = FingerprintPreprocessor(device=self.device)
        except TypeError:
            self.preprocessor = FingerprintPreprocessor()

        try:
            self.feature_extractor = FingerprintFeatureExtractor(device=self.device)
        except TypeError:
            self.feature_extractor = FingerprintFeatureExtractor()

        try:
            self.matcher = FingerprintMatcher(device=self.device)
        except TypeError:
            self.matcher = FingerprintMatcher()

        self.logger.info("Fingerprint matching pipeline initialized.")

    def process_single_fingerprint(self, image_path):
        """Process a single fingerprint image"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            # Preprocessing
            processed_images = self.preprocessor.process_fingerprint(image_path)

            # Feature extraction
            minutiae = self.feature_extractor.extract_minutiae(
                processed_images["skeleton"]
            )
            orb_kp, orb_desc = self.feature_extractor.extract_orb_features(
                processed_images["enhanced"]
            )

            return {
                "images": processed_images,
                "minutiae": minutiae,
                "orb_keypoints": orb_kp,
                "orb_descriptors": orb_desc,
                "path": image_path,
            }

        except Exception as e:
            self.logger.error(f"Error processing fingerprint {image_path}: {e}")
            raise

    def match_fingerprints(self, image_path1, image_path2):
        """Complete fingerprint matching pipeline"""
        try:
            self.logger.info(
                f"Starting fingerprint matching: {image_path1} vs {image_path2}"
            )

            # Process fingerprints
            fp1 = self.process_single_fingerprint(image_path1)
            fp2 = self.process_single_fingerprint(image_path2)

            # Match features
            minutiae_score, minutiae_matches = self.matcher.match_minutiae(
                fp1["minutiae"], fp2["minutiae"]
            )

            orb_score, orb_matches = self.matcher.match_orb_features(
                fp1["orb_descriptors"], fp2["orb_descriptors"]
            )

            overall_score = self.matcher.calculate_overall_similarity(
                minutiae_score, orb_score
            )
            is_match = overall_score > self.matcher.threshold

            result = {
                "is_match": is_match,
                "overall_score": overall_score,
                "minutiae_score": minutiae_score,
                "orb_score": orb_score,
                "minutiae_matches": len(minutiae_matches),
                "orb_matches": len(orb_matches),
                "fingerprint1": fp1,
                "fingerprint2": fp2,
            }

            self.logger.info(
                f"Matching complete. Overall score: {overall_score:.3f}, Match: {is_match}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in fingerprint matching pipeline: {e}")
            raise