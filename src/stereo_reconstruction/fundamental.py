"""
"""

from typing import Tuple, Sequence

from abc import ABC, abstractmethod
import argparse

import numpy as np
import numpy.typing as npt
import cv2


class Sampler(ABC):
    def __init__(self):
        self.rng = np.random.default_rng()

    @abstractmethod
    def sample(self):
        pass


class EightCorrespondenceSampler(Sampler):
    def __init__(self, sample_size=8):
        super().__init__()
        self.sample_size = sample_size

    def sample(self, inputs: Sequence[npt.NDArray[np.int32]]):
        x1, x2 = inputs
        in_indexes = self.rng.choice(x1.shape[0], self.sample_size, replace=False)
        out_indexes = [index for index in range(x1.shape[0]) if index not in in_indexes]

        inset = (x1[in_indexes], x2[in_indexes])
        outset = (x1[out_indexes], x2[out_indexes])

        return (inset, outset)


def normalize(pts: npt.NDArray[np.int32]) -> Tuple[npt.NDArray[np.int32], int]:
    """
    Normalize the pts using the strategy outline in Hartley 1997
    https://mil.ufl.edu/nechyba/www/eel6562/course_materials/t9.3d_vision/hartley1997.pdf

    x' [3x1] = T[3x3] x[3x1]

    hpts = x'
    """
    centroid = pts.mean(axis=1, keepdims=True)
    scale = np.mean(np.linalg.norm(pts-centroid, axis=0))
    hpts = np.vstack((pts, np.ones((1, pts.shape[1]))))  # homogeneous coords
    translation = np.array(((0, 0, -centroid[0, 0]*np.sqrt(2)/scale), (0, 0, -centroid[1, 0]*np.sqrt(2)/scale), (0, 0, 0)))
    scaling = np.array(((np.sqrt(2)/scale, 0, 0), (0, np.sqrt(2)/scale, 0), (0, 0, 1)))
    normalization = scaling + translation

    hpts = normalization @ hpts

    return hpts, normalization


def get_fundamental_matrix(pts1, pts2):
    """
    Here be dragons
    """
    pts1 = pts1.T
    pts2 = pts2.T

    npts1, T1 = normalize(pts1)
    npts2, T2 = normalize(pts2)

    u1 = npts1[0, :].reshape(-1, 1)
    v1 = npts1[1, :].reshape(-1, 1)
    u2 = npts2[0, :].reshape(-1, 1)
    v2 = npts2[1, :].reshape(-1, 1)
    ones = np.ones_like(u1)

    A = np.hstack((u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, ones))

    U, D, Vt = np.linalg.svd(A, full_matrices=True)

    V = Vt.T
    f = V[:, -1]
    f11, f21, f31, f12, f22, f32, f13, f23, f33 = f
    Fa = np.array(((f11, f12, f13), (f21, f22, f23), (f31, f32, f33)))

    Ua, Da, Vta = np.linalg.svd(Fa, full_matrices=True)
    Ft = Ua @ np.diag((Da[0], Da[1], 0)) @ Vta

    F = T2.T @ Ft @ T1
    F = F / F[-1, -1]

    return F


def get_ZW():
    #  https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&index=48&t=3198s
    Z = np.array(((0, 1, 0),
                  (-1, 0, 0),
                  (0, 0, 0)))
    W = np.array(((0, -1, 0),
                  (1, 0, 0),
                  (0, 0, 1)))
    yield Z, W
    yield -Z.T, W
    yield -Z, W.T
    yield Z.T, W.T


def get_essential_matrix(pts1, pts2, K1=np.eye(3), K2=np.eye(3)):
    """
    Here be dragons
    """
    pts1 = pts1.T
    pts2 = pts2.T

    kpts1 = (np.linalg.inv(K1) @ np.vstack((pts1, np.ones((1, pts1.shape[1])))))[:2, :]  # Double check but shouldn't need to divide by vector[-1]
    kpts2 = (np.linalg.inv(K2) @ np.vstack((pts2, np.ones((1, pts2.shape[1])))))[:2, :]

    npts1, T1 = normalize(kpts1)
    npts2, T2 = normalize(kpts2)

    u1 = npts1[0, :].reshape(-1, 1)
    v1 = npts1[1, :].reshape(-1, 1)
    u2 = npts2[0, :].reshape(-1, 1)
    v2 = npts2[1, :].reshape(-1, 1)
    ones = np.ones_like(u1)

    A = np.hstack((u1*u2, u1*v2, u1, v1*u2, v1*v2, v1, u2, v2, ones))

    U, D, Vt = np.linalg.svd(A, full_matrices=True)

    V = Vt.T
    e = V[:, -1]
    e11, e21, e31, e12, e22, e32, e13, e23, e33 = e
    Ea = np.array(((e11, e12, e13), (e21, e22, e23), (e31, e32, e33)))

    Ua, Da, Vta = np.linalg.svd(Ea, full_matrices=True)
    Et = Ua @ np.diag((1, 1, 0)) @ Vta

    E = T2.T @ Et @ T1
    E = E / E[-1, -1]

    return E


def get_baseline(E):
    U, D, Vt = np.linalg.svd(E, full_matrices=True)

    # Ensure D[1] = D[2] = 1
    Dc = np.diag((1,1,0))
    Vtc = Vt.copy()
    Vtc[0] = Vtc[0] * D[0]
    Vtc[1] = Vtc[1] * D[1]

    ZW_generator = get_ZW()

    for Z, W in ZW_generator:
        E = U @ Z @ U.T @ U @ W @ Vt
        Sb = U @ Z @ U.T
        R = U @ W @ Vt

    print('stall')


def fundamental_equality(F, correspondences):
    x1, x2 = correspondences
    hx1 = np.hstack((x1, np.ones((x1.shape[0], 1)))).T
    hx2 = np.hstack((x2, np.ones((x2.shape[0], 1)))).T
    return np.diagonal(hx2.T @ F @ hx1)


def RANSAC(function, test_function, inputs, sampler: Sampler, iterations: int, threshold: float):
    """
    https://vitalflux.com/ransac-regression-explained-with-python-examples/#RANSAC_Regression_Algorithm_Details
    """
    max_inlier = -float("inf")
    final_output = None
    final_output_set = None
    for _ in range(iterations):
        in_sample, out_sample = sampler.sample(inputs)  # Step 1
        output = function(*in_sample)  # Step 1
        mask = np.abs(test_function(output, out_sample)) < threshold  # Step 2 + 3
        new_ins = tuple([np.vstack((in_samp, out_samp[mask])) for in_samp, out_samp in zip(in_sample, out_sample)])  # Step 4
        new_output = function(*new_ins)  # Step 4
        mask = np.abs(test_function(new_output, new_ins)) < threshold  # Step 5
        if np.sum(mask) > max_inlier:
            max_inlier = np.sum(mask)
            final_output = new_output
            final_output_set = tuple([new_in[mask] for new_in in new_ins])
    return final_output, final_output_set


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--UseRANSAC", type=int, default=0)
    parser.add_argument("--image1", type=str, default='data/towerLeft.jpg')
    parser.add_argument("--image2", type=str, default='data/towerRight.jpg')
    args = parser.parse_args()

    img1 = cv2.imread(args.image1, 0)
    img2 = cv2.imread(args.image2, 0)

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
        if m.distance < 0.8 * n.distance:
            good.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # plot matching points
    cv_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts1]
    cv_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts2]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(cv_kp1))]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    sampler = EightCorrespondenceSampler(sample_size=8)

    # F, inliers = RANSAC(function=get_essential_matrix,
    F, inliers = RANSAC(function=get_essential_matrix,
                        test_function=fundamental_equality,
                        inputs=(pts1, pts2),
                        sampler=sampler,
                        iterations=1000,
                        threshold=0.2)

    print(F)
    print(f'Percentage of inlier points / total input points {inliers[0].shape[0]} / {pts1.shape[0]} = {inliers[0].shape[0] / pts1.shape[0]:.1%}.')

    # E = K'T F K

    get_baseline(F)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
