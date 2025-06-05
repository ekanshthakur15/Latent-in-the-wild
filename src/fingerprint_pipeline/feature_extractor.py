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
            minutiae = []
            kernel = np.ones((3, 3), np.uint8)
            binary_skeleton = (skeleton_image > 127).astype(np.uint8)
            height, width = binary_skeleton.shape

            for i in range(1, height - 1):
                for j in range(1, width - 1):
                    if binary_skeleton[i, j] == 1:
                        neighbors = binary_skeleton[i - 1 : i + 2, j - 1 : j + 2]
                        neighbor_count = np.sum(neighbors) - 1

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
            if (
                y > 0
                and y < binary_image.shape[0] - 1
                and x > 0
                and x < binary_image.shape[1] - 1
            ):
                # Cast to int to avoid overflow in subtraction
                dy = int(binary_image[y + 1, x]) - int(binary_image[y - 1, x])
                dx = int(binary_image[y, x + 1]) - int(binary_image[y, x - 1])
                angle = np.arctan2(dy, dx)
                return angle
            return 0.0
        except Exception:
            return 0.0

    def extract_orb_features(self, image):
        """Extract ORB features with GPU acceleration if available"""
        try:
            use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

            if use_cuda:
                self.logger.info("Using CUDA-accelerated ORB")

                # Upload image to GPU
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)

                # Convert to grayscale if needed
                if len(image.shape) == 3:
                    gpu_image = cv2.cuda.cvtColor(gpu_image, cv2.COLOR_BGR2GRAY)

                # Create ORB detector (CUDA version)
                orb = cv2.cuda_ORB.create(nfeatures=500)

                # Detect and compute features on GPU
                keypoints_gpu, descriptors = orb.detectAndComputeAsync(gpu_image, None)
                keypoints = orb.convert(keypoints_gpu)

                return keypoints, descriptors

            else:
                self.logger.info("Using CPU ORB")

                orb = cv2.ORB_create(nfeatures=500)
                keypoints, descriptors = orb.detectAndCompute(image, None)
                return keypoints, descriptors

        except Exception as e:
            self.logger.error(f"Error extracting ORB features: {e}")
            return [], None