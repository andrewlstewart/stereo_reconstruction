"""
Script to generate depth maps given stereo images and camera parameters
"""

import argparse
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Used in the projection='3d'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera_parameter_path', type=str, required=True, help="Location of the numpy arrays for the intrinsic and extrinsic camera parameters.")
    parser.add_argument("--stereo_images_path", type=str, required=True, help="Location of the stereo images to create depth estimations.")
    parser.add_argument("--image_extension", type=str, default='.jpg')
    args = parser.parse_args()
    return args


def get_correspondencies(img1, img2):
    # in future, use more sophisticated method using epipolar lines to get more correspondencies
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    pts1 = []
    pts2 = []

    # ratio test as per Lowe's paper
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3, 0.99)

    return pts1[mask[:,0].astype(bool)], pts2[mask[:,0].astype(bool)]


def compute_direct_linear_transformation(projection1, projection2, pt1, pt2):
    # https://temugeb.github.io/computer_vision/2021/02/06/direct-linear-transorms.html
    # Parallel vectors, therefore cross product is zero (sin(0) = 0)
    # v1p3 - p2 ; p1 - u1p3 ; u1p2 - v1p1 <- linear combination of the first 2
    # We can stack both points because we're just trying to solve a set of simultaneous eq @ X = 0
    A = np.array([pt1[1] * projection1[2, :] - projection1[1, :],
                  projection1[0, :] - pt1[0] * projection1[2, :],
                  pt2[1] * projection2[2, :] - projection2[1, :],
                  projection2[0, :] - pt2[0] * projection2[2, :]])
    B = A.T @ A
    U, S, Vt = np.linalg.svd(B, full_matrices = True)
    return Vt[3, 0:3] / Vt[3, 3]


def main() -> int:
    args = parse_args()

    image_size, K1, distortion1, K2, distortion2, rotation, translation = [np.load(os.path.join(args.camera_parameter_path, f"{file_name}.npy")) for file_name in ["image_size", "K1", "distortion1", "K2", "distortion2", "rotation", "translation"]]

    for root, dirs, files in os.walk(args.stereo_images_path):
        for file_ in files:
            if os.path.splitext(file_)[1] != args.image_extension:
                 continue
            
            image = cv2.imread(os.path.join(root, file_))
            left_image = image[:, :2028, :]
            right_image = image[:, 2028:, :]

            pts1, pts2 = get_correspondencies(left_image, right_image)

            X = np.zeros((len(pts1), 3))

            # Left camera = reference, therefore rotation = identity and translation = 0
            RT1 = np.hstack([np.eye(3), [[0],[0],[0]]])
            P1 = K1 @ RT1

            RT2 = np.hstack([rotation, translation])
            P2 = K2 @ RT2

            for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
                X[idx] = compute_direct_linear_transformation(P1, P2, pt1, pt2)
 
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(131)
            ax.imshow(left_image)
            ax.scatter(pts1[:, 0], pts1[:, 1])
            ax = fig.add_subplot(132)
            ax.imshow(right_image)
            ax.scatter(pts2[:, 0], pts2[:, 1])
            ax = fig.add_subplot(133, projection='3d')
            ax.scatter(xs = X[:, 0], ys = X[:, 1], zs = X[:, 2])
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
