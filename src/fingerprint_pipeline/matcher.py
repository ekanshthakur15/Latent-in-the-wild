import cv2
import numpy as np
from scipy.spatial.distance import cdist
import logging


class FingerprintMatcher:
    """Match fingerprint features and compute similarity scores"""

    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def match_minutiae(self, minutiae1, minutiae2):
        """Match minutiae points between two fingerprints"""
        try:
            if not minutiae1 or not minutiae2:
                return 0.0, []

            # Extract coordinates
            coords1 = np.array([(m["x"], m["y"]) for m in minutiae1])
            coords2 = np.array([(m["x"], m["y"]) for m in minutiae2])

            # Calculate distance matrix
            distances = cdist(coords1, coords2, metric="euclidean")

            # Find matches based on distance threshold
            matches = []
            distance_threshold = 20  # pixels

            for i in range(len(minutiae1)):
                min_dist_idx = np.argmin(distances[i])
                min_dist = distances[i][min_dist_idx]

                if min_dist < distance_threshold:
                    # Check angle similarity
                    angle_diff = abs(
                        minutiae1[i]["angle"] - minutiae2[min_dist_idx]["angle"]
                    )
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Wrap around

                    if angle_diff < np.pi / 4:  # 45 degrees threshold
                        matches.append(
                            {
                                "point1": i,
                                "point2": min_dist_idx,
                                "distance": min_dist,
                                "angle_diff": angle_diff,
                            }
                        )

                        # Mark as used to avoid multiple matches
                        distances[:, min_dist_idx] = float("inf")

            # Calculate similarity score
            similarity = len(matches) / max(len(minutiae1), len(minutiae2))

            self.logger.info(
                f"Found {len(matches)} minutiae matches, similarity: {similarity:.3f}"
            )
            return similarity, matches

        except Exception as e:
            self.logger.error(f"Error matching minutiae: {e}")
            return 0.0, []

    def match_orb_features(self, desc1, desc2):
        """Match ORB descriptors"""
        try:
            if desc1 is None or desc2 is None:
                return 0.0, []

            # Use FLANN matcher for better performance
            FLANN_INDEX_LSH = 6
            index_params = dict(
                algorithm=FLANN_INDEX_LSH,
                table_number=6,
                key_size=12,
                multi_probe_level=1,
            )
            search_params = dict(checks=50)

            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            # Calculate similarity
            similarity = len(good_matches) / max(len(desc1), len(desc2))

            self.logger.info(
                f"Found {len(good_matches)} ORB matches, similarity: {similarity:.3f}"
            )
            return similarity, good_matches

        except Exception as e:
            self.logger.error(f"Error matching ORB features: {e}")
            return 0.0, []

    def calculate_overall_similarity(
        self, minutiae_score, orb_score, weights=(0.7, 0.3)
    ):
        """Calculate weighted overall similarity score"""
        try:
            overall_score = weights[0] * minutiae_score + weights[1] * orb_score
            return overall_score

        except Exception as e:
            self.logger.error(f"Error calculating overall similarity: {e}")
            return 0.0
