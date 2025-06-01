import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, filters
from skimage.restoration import denoise_nl_means
import logging


class FingerprintPreprocessor:
    """Handles fingerprint image preprocessing including denoising, enhancement, and skeletonization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def enhance_image(self, image):
        """Apply comprehensive image enhancement pipeline"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Normalize image
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Apply non-local means denoising
            denoised = denoise_nl_means(
                image, h=10, fast_mode=True, patch_size=7, patch_distance=11
            )
            denoised = (denoised * 255).astype(np.uint8)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)

            # Apply histogram equalization
            equalized = cv2.equalizeHist(blurred)

            # Apply Gabor filter bank for ridge enhancement
            enhanced = self._apply_gabor_filters(equalized)

            return enhanced

        except Exception as e:
            self.logger.error(f"Error in image enhancement: {e}")
            return image

    def _apply_gabor_filters(self, image):
        """Apply Gabor filters for ridge enhancement"""
        try:
            # Create Gabor filter bank
            filters_bank = []
            angles = [0, 45, 90, 135]  # Different orientations

            for angle in angles:
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    5,
                    np.radians(angle),
                    2 * np.pi * 0.1,
                    0.5,
                    0,
                    ktype=cv2.CV_32F,
                )
                filters_bank.append(kernel)

            # Apply filters and combine responses
            responses = []
            for kernel in filters_bank:
                filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
                responses.append(filtered)

            # Take maximum response
            enhanced = np.maximum.reduce(responses)
            return enhanced

        except Exception as e:
            self.logger.error(f"Error in Gabor filtering: {e}")
            return image

    def binarize_image(self, image):
        """Convert image to binary using adaptive thresholding"""
        try:
            # Apply adaptive threshold
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5
            )

            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            return binary

        except Exception as e:
            self.logger.error(f"Error in binarization: {e}")
            return image

    def skeletonize_image(self, binary_image):
        """Extract skeleton of fingerprint ridges"""
        try:
            # Ensure binary image (0 and 1 values)
            binary = (binary_image > 127).astype(np.uint8)

            # Apply morphological skeletonization
            skeleton = morphology.skeletonize(binary)

            # Convert back to 0-255 range
            skeleton = (skeleton * 255).astype(np.uint8)

            return skeleton

        except Exception as e:
            self.logger.error(f"Error in skeletonization: {e}")
            return binary_image

    def process_fingerprint(self, image_path):
        """Complete preprocessing pipeline"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            self.logger.info(f"Processing fingerprint: {image_path}")

            # Enhancement pipeline
            enhanced = self.enhance_image(image)
            binary = self.binarize_image(enhanced)
            skeleton = self.skeletonize_image(binary)

            return {
                "original": image,
                "enhanced": enhanced,
                "binary": binary,
                "skeleton": skeleton,
            }

        except Exception as e:
            self.logger.error(f"Error processing fingerprint {image_path}: {e}")
            raise
