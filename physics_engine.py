#!/usr/bin/env python
"""
Physics Engine Module: Physical Hardware & Sensor Degradation Functions
Simulating edge sensor reality: Gaussian noise, Salt & Pepper, Poisson,
JPEG Lossy Compression, Biomechanical Elastic Tissue Deformation, and ROI Perturbation.

Author: Lead Biomedical ML Engineer (Q1 Journal Submission Suite)
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates

RANDOM_STATE = 42


def apply_gaussian_noise(imgs, sigma=0.03):
    """
    Applies additive zero-mean Gaussian noise N(0, sigma^2) to image batch.
    Clips output to valid [0, 1] range.
    """
    np.random.seed(RANDOM_STATE)
    noise = np.random.normal(0, sigma, imgs.shape).astype(np.float32)
    noisy_imgs = np.clip(imgs + noise, 0.0, 1.0)
    return noisy_imgs


def apply_salt_pepper_noise(imgs, amount=0.03):
    """
    Applies Salt & Pepper noise to image batch.
    Fraction 'amount/2' set to 0 (pepper), fraction 'amount/2' set to 1 (salt).
    """
    np.random.seed(RANDOM_STATE)
    noisy_imgs = imgs.copy()
    num_pixels = imgs.size
    num_noise = int(amount * num_pixels)

    flat_indices = np.random.choice(num_pixels, num_noise, replace=False)
    salt_idx = flat_indices[:num_noise // 2]
    pepper_idx = flat_indices[num_noise // 2:]

    flat_imgs = noisy_imgs.ravel()
    flat_imgs[salt_idx] = 1.0
    flat_imgs[pepper_idx] = 0.0

    return noisy_imgs.reshape(imgs.shape)


def apply_poisson_noise(imgs):
    """
    Applies Poisson noise (shot noise) to image batch.
    """
    np.random.seed(RANDOM_STATE)
    # Scale up to photon count representation
    scaled = imgs * 255.0
    # Avoid zero or negative values
    scaled = np.maximum(scaled, 0.0)
    noisy = np.random.poisson(scaled).astype(np.float32)
    noisy = np.clip(noisy / 255.0, 0.0, 1.0)
    return noisy


def apply_jpeg_compression(imgs, quality=70):
    """
    Applies JPEG lossy compression artifacts to image batch using OpenCV.
    Quality range: [1, 100]
    """
    compressed_imgs = np.zeros_like(imgs)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]

    for i in range(len(imgs)):
        img_8u = (imgs[i] * 255.0).astype(np.uint8)
        _, encoded = cv2.imencode('.jpg', img_8u, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        compressed_imgs[i] = decoded.astype(np.float32) / 255.0

    return compressed_imgs


def apply_elastic_deformation(imgs, alpha=10.0, sigma=3.0):
    """
    Applies biomechanical elastic tissue deformation using scipy.ndimage.map_coordinates
    with smooth Gaussian-filtered random displacement fields.
    Simulates physical compression of breast tissue during mammography/ultrasound.
    """
    if alpha == 0:
        return imgs.copy()

    deformed_imgs = np.zeros_like(imgs)
    h, w = imgs.shape[1], imgs.shape[2]

    for i in range(len(imgs)):
        img = imgs[i]
        np.random.seed(RANDOM_STATE + i)

        dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(w), np.arange(h))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        distorted_img = map_coordinates(img, indices, order=1, mode='reflect')
        deformed_imgs[i] = distorted_img.reshape(h, w)

    return np.clip(deformed_imgs, 0.0, 1.0)


def apply_roi_perturbation(orig_imgs, roi_boxes, perturb_pct=0.10, target_size=(64, 64)):
    """
    Applies bounding box shift and scale perturbations to ROI bounding boxes
    before cropping from original full-size grayscale images.
    Simulates physical sensor alignment error or automated segmentation jitter.
    """
    if perturb_pct == 0:
        perturbed_imgs = []
        for img, (x, y, w, h) in zip(orig_imgs, roi_boxes):
            crop = img[y:y+h, x:x+w] if w > 5 and h > 5 else img
            resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
            perturbed_imgs.append(resized.astype(np.float32) / 255.0)
        return np.array(perturbed_imgs)

    perturbed_imgs = []
    np.random.seed(RANDOM_STATE)

    for img, (x, y, w, h) in zip(orig_imgs, roi_boxes):
        img_h, img_w = img.shape[:2]

        # Shift offsets
        dx = int(np.random.uniform(-perturb_pct, perturb_pct) * w)
        dy = int(np.random.uniform(-perturb_pct, perturb_pct) * h)

        # Scale offsets
        dw = int(np.random.uniform(-perturb_pct, perturb_pct) * w)
        dh = int(np.random.uniform(-perturb_pct, perturb_pct) * h)

        new_x = max(0, min(img_w - 10, x + dx))
        new_y = max(0, min(img_h - 10, y + dy))
        new_w = max(10, min(img_w - new_x, w + dw))
        new_h = max(10, min(img_h - new_y, h + dh))

        crop = img[new_y:new_y+new_h, new_x:new_x+new_w]
        resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        perturbed_imgs.append(resized.astype(np.float32) / 255.0)

    return np.array(perturbed_imgs)
