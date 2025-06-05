import cv2
import numpy as np
from skimage import morphology
import logging



class FingerprintPreprocessor:
    """Handles fingerprint image preprocessing including denoising, enhancement, and skeletonization"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0

    def enhance_image(self, image):
        """Apply comprehensive image enhancement pipeline"""
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            if self.cuda_available:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)
                gpu_denoised = cv2.cuda_GaussianBlur(gpu_image, (7, 7), 0).download()
                denoised = gpu_denoised.astype(np.uint8)
            else:
                # CUDA not available. Falling back to CPU GaussianBlur.
                denoised = cv2.GaussianBlur(image, (7, 7), 0)

            blurred = cv2.GaussianBlur(denoised, (3, 3), 0)
            equalized = cv2.equalizeHist(blurred)
            enhanced = self._apply_gabor_filters(equalized)
            return enhanced

        except Exception as e:
            self.logger.error(f"Error in image enhancement: {e}")
            return image

    def _apply_gabor_filters(self, image):
        """Apply Gabor filters for ridge enhancement"""
        try:
            filters_bank = []
            angles = [0, 45, 90, 135]

            for angle in angles:
                kernel = cv2.getGaborKernel(
                    (21, 21), 5, np.radians(angle), 2 * np.pi * 0.1, 0.5, 0, ktype=cv2.CV_32F
                )
                filters_bank.append(kernel)

            responses = [cv2.filter2D(image, cv2.CV_8UC3, kernel) for kernel in filters_bank]
            enhanced = np.maximum.reduce(responses)
            return enhanced

        except Exception as e:
            self.logger.error(f"Error in Gabor filtering: {e}")
            return image

    def binarize_image(self, image):
        """Convert image to binary using adaptive thresholding"""
        try:
            if self.cuda_available:
                gpu_image = cv2.cuda_GpuMat()
                gpu_image.upload(image)

                gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (15, 15), 0)
                blurred = gpu_blur.apply(gpu_image).download()
            else:
                # CUDA not available. Falling back to CPU GaussianBlur.
                blurred = cv2.GaussianBlur(image, (15, 15), 0)

            _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

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
            binary = (binary_image > 127).astype(np.uint8)
            skeleton = morphology.skeletonize(binary)
            return (skeleton * 255).astype(np.uint8)
        except Exception as e:
            self.logger.error(f"Error in skeletonization: {e}")
            return binary_image

    def process_fingerprint(self, image_path):
        """Complete preprocessing pipeline"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            self.logger.info(f"Processing fingerprint: {image_path}")

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