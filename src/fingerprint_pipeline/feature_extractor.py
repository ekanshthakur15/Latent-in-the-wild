import cv2
import numpy as np
from scipy.spatial.distance import cdist
import logging


class FingerprintFeatureExtractor:
    """Extract and match fingerprint features (minutiae points)"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_minutiae(self, skeleton_image):
        """Extract minutiae points from skeleton image"""
        try:
            # Find ridge endings and bifurcations
            minutiae = []

            # Create 3x3 kernel for checking neighbors
            kernel = np.ones((3, 3), np.uint8)

            # Convert to binary (0 and 1)
            binary_skeleton = (skeleton_image > 127).astype(np.uint8)

            height, width = binary_skeleton.shape

            # Scan for minutiae points
            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if binary_skeleton[i, j] == 1:  # Ridge pixel
                        # Count neighbors
                        neighbors = binary_skeleton[i - 1 : i + 2, j - 1 : j + 2]
                        neighbor_count = np.sum(neighbors) - 1  # Exclude center pixel

                        # Ridge ending (1 neighbor)
                        if neighbor_count == 1:
                            minutiae.append(
                                {
                                    "x": j,
                                    "y": i,
                                    "type": "ending",
                                    "angle": self._calculate_ridge_direction(
                                        binary_skeleton, i, j
                                    ),
                                }
                            )

                        # Bifurcation (3 or more neighbors)
                        elif neighbor_count >= 3:
                            minutiae.append(
                                {
                                    "x": j,
                                    "y": i,
                                    "type": "bifurcation",
                                    "angle": self._calculate_ridge_direction(
                                        binary_skeleton, i, j
                                    ),
                                }
                            )

            self.logger.info(f"Extracted {len(minutiae)} minutiae points")
            return minutiae

        except Exception as e:
            self.logger.error(f"Error extracting minutiae: {e}")
            return []

    def _calculate_ridge_direction(self, binary_image, y, x):
        """Calculate ridge direction at given point"""
        try:
            # Simple gradient-based direction calculation
            if (
                y > 0
                and y < binary_image.shape[0] - 1
                and x > 0
                and x < binary_image.shape[1] - 1
            ):
                dy = binary_image[y + 1, x] - binary_image[y - 1, x]
                dx = binary_image[y, x + 1] - binary_image[y, x - 1]
                angle = np.arctan2(dy, dx)
                return angle
            return 0.0

        except Exception as e:
            return 0.0

    def extract_orb_features(self, image):
        """Extract ORB features as backup method"""
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=500)

            # Find keypoints and descriptors
            keypoints, descriptors = orb.detectAndCompute(image, None)

            return keypoints, descriptors

        except Exception as e:
            self.logger.error(f"Error extracting ORB features: {e}")
            return [], None
