"""
An implementation of Hartley's 8-point algorithm to compute the fundamental matrix, then an 8-point algorithm to compute
the essential matrix + decomposition into a translation and rotation to get the baseline between the stereo cameras.

Much of this implementation follows the course given by Professor Cyrill Stachniss's and his lectures:
    https://www.youtube.com/watch?v=uHApDqH-8UE&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y
    and
    https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y

No effort has been made to account for distortions.

This code is mostly just an exercise to understand some deeper computer vision concepts.

In practice, you should probably just call cv2.calibrateCamera https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
then cv2.findEssentialMat or cv2.findFundamentalMat.
"""

from typing import Tuple, Sequence, Optional

from abc import ABC, abstractmethod
import argparse
from pathlib import Path
import itertools

import numpy as np
import numpy.typing as npt
import cv2


class Sampler(ABC):
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng()
        if seed:
            self.rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def sample(self):
        pass


class EightCorrespondenceSampler(Sampler):
    """
    Simple class to with a sample method which samples points from a input and returns the sample points as the inset and the non-sampled points as the outset
    """

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


def get_fundamental_matrix(pts1: npt.NDArray[np.int32], pts2: npt.NDArray[np.int32]):
    """
    Here be dragons.  Refer to Hartley for clear notation.
    http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj_final/www/amichals/fundamental.pdf

    Solves the coplanarity constrait in real coordinates.

    up.T @ F @ u
    """
    pts1 = pts1.T
    pts2 = pts2.T

    npts1, T1 = normalize(pts1)
    npts2, T2 = normalize(pts2)

    u = npts1[0, :].reshape(-1, 1)
    v = npts1[1, :].reshape(-1, 1)
    up = npts2[0, :].reshape(-1, 1)
    vp = npts2[1, :].reshape(-1, 1)
    ones = np.ones_like(u)

    A = np.hstack((u*up, u*vp, u, v*up, v*vp, v, up, vp, ones))

    U, D, Vt = np.linalg.svd(A, full_matrices=True)

    V = Vt.T
    f = V[:, -1]
    f11, f21, f31, f12, f22, f32, f13, f23, f33 = f
    Fa = np.array(((f11, f12, f13), (f21, f22, f23), (f31, f32, f33)))

    Ua, Da, Vta = np.linalg.svd(Fa, full_matrices=True)
    Ft = Ua @ np.diag((Da[0], Da[1], 0)) @ Vta

    F = T2.T @ Ft @ T1
    F = F / F[-1, -1]  # TODO: Investigate if this is required.  To get 'matching' cv2 values, they normalize.

    return F


def get_essential_matrix(npts1: npt.NDArray[np.float32],
                         npts2: npt.NDArray[np.float32],
                         c: float,
                         cp: float) -> npt.NDArray[np.float32]:
    """
    Solving the coplanarity constraint for calibrated cameras.  Therefore npoints need to be in calibrated space. npts = np.linalg.inv(K) @ pts

    (y').T @ E @ y = 0

    https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=1982s
    https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=2197s

        I'm kind of convinced that normalization should <i>not</i> be done for the essential matrix.  If it is done, then either E or E_hat
        satisfy U @ diag(1,1,0) @ V.T but if E satisfies that condition, then U and V.T are no longer rotation matrices (T.T @ U is not a rotation matrix)
        and therefore it can't be decomposed.  Therefore, don't normalize, then E = U @ diag(1,1,0) @ V.T without any tranformation matrix and U is a rotation matrix.
        I reached out to Professor Stachniss directly but I didn't get a response.  If I do hear differently, I will update the code and make a reference.

        https://stackoverflow.com/a/34495431
    """

    npts1 = npts1.T
    npts2 = npts2.T

    u = npts1[0, :].reshape(-1, 1)
    v = npts1[1, :].reshape(-1, 1)
    up = npts2[0, :].reshape(-1, 1)
    vp = npts2[1, :].reshape(-1, 1)
    c_vec = np.ones_like(u) * c
    cp_vec = np.ones_like(up) * cp

    A = np.hstack((u*up, u*vp, u*cp_vec, v*up, v*vp, v*cp_vec, up*c_vec, vp*c_vec, c_vec*cp_vec))

    U, D, Vt = np.linalg.svd(A, full_matrices=True)

    V = Vt.T
    e = V[:, -1]
    e11, e21, e31, e12, e22, e32, e13, e23, e33 = e
    Ea = np.array(((e11, e12, e13), (e21, e22, e23), (e31, e32, e33)))

    Ua, Da, Vta = np.linalg.svd(Ea, full_matrices=True)
    E = Ua @ np.diag((1, 1, 0)) @ Vta

    return E


def get_baseline(E: npt.NDArray[np.float32],
                 calibrated_pts1: npt.NDArray[np.float32],
                 calibrated_pts2: npt.NDArray[np.float32],
                 c: float,
                 cp: float) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Get out the rotation and the basis vector from the essential matrix.

    https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=2624s
    """
    h_pts1 = np.hstack((calibrated_pts1, np.ones((calibrated_pts1.shape[0], 1))*c)).T
    h_pts2 = np.hstack((calibrated_pts1, np.ones((calibrated_pts2.shape[0], 1))*cp)).T

    U, D, Vt = np.linalg.svd(E, full_matrices=True)

    # Ensure D[1] = D[2] = 1
    assert np.isclose(D[0], D[1])  # Sb will be up to scale, so the first two diagonal elements can not equal to 1 but have to be the same
    assert np.isclose(D[2], 0)

    best_answer = -float('inf')
    best_truths = False

    #  https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=3198s
    Z = np.array(((0, 1, 0),
                  (-1, 0, 0),
                  (0, 0, 0)))
    W = np.array(((0, -1, 0),
                  (1, 0, 0),
                  (0, 0, 1)))

    P = np.hstack((np.eye(3), np.zeros(3).reshape(-1, 1)))
    P1T = P[0, :]
    P2T = P[1, :]
    P3T = P[2, :]
    for W_, t in itertools.product((W, W.T), (U[:, -1].reshape(-1, 1), -U[:, -1].reshape(-1, 1))):
        Pp = np.hstack((U @ W_ @ Vt, t))
        Pp1T = Pp[0, :]
        Pp2T = Pp[1, :]
        Pp3T = Pp[2, :]

        As = []
        for ((x, y), (xp, yp)) in zip(calibrated_pts1, calibrated_pts2):
            # http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf
            # Hartley & Zisserman, Multiple View Geometry in computer vision Second Edition Cambridge, page 259, 
            # 312-313 homogeneous method
            A = np.vstack((x*P3T - P1T,
                           y*P3T - P2T,
                           xp*Pp3T - Pp1T,
                           yp*Pp3T - Pp2T
                           ))
            As.append(A)
        As = np.vstack(As)

        U_, D_, Vt_ = np.linalg.svd(As, full_matrices=True)
        
        X = Vt_[3,:] / Vt_[3,3]  # What is this actually computing, some mass center point of all points?

        X = X[:3].reshape(-1, 1)

        if X[2] < 0:
            continue
        
        # angle (cosine) between x, y, z and projected point on image plane
        R = Pp[:, :3]
        t = Pp[:, 3]

        Xi = R @ ((np.array((0, 0, 1)) - t).reshape(-1, 1))

        angle = np.arccos(np.dot(X[:,0], Xi[:,0]) / (np.linalg.norm(Xi)*np.linalg.norm(X)))

        if angle < np.pi/2:
            return R, t
        
    raise Exception()

    # https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=55m50s
    Sb1 = U @ Z @ U.T
    Sb2 = U @ Z.T @ U.T
    R1T = U @ W @ Vt
    R2T = U @ W.T @ Vt

    for Sb, Rt in itertools.product((Sb1, Sb2), (R1T, R2T)):
        R = Rt.T
        t = np.array((-Sb[1, 2], Sb[0, 2], -Sb[0, 1])).reshape(-1, 1)

        # https://www.youtube.com/watch?v=UZlRhEUWSas&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=330s
        truths = []
        for pt_1, pt_2 in zip(h_pts1.T, h_pts2.T):

            p = np.array((0, 0, 0)).reshape((3, 1))
            q = t[:, 0].reshape((3, 1))

            r = pt_1.reshape((3, 1))  # R'.T = I.T = I
            # The calibrated point after rotation needs to be shifted by the baseline as well. This isn't brought up in the lecture though.
            s = ((R.T @ pt_2) + t[:, 0]).reshape((3, 1))

            A = np.array((((r.T @ r)[0, 0], (-s.T @ r)[0, 0]), ((r.T @ s)[0, 0], (-s.T @ s)[0, 0])))
            b = np.vstack(((q - p).T @ r, (q - p).T @ s))  # The professors slide 13 is incorrect

            x = np.linalg.inv(A) @ b

            # lam = x[0]
            # mu = x[1]

            # F = p + lam * r
            # G = q + mu * s

            # H = (F+G)/2

            truths.append(np.all(x > 0))

        if sum(truths) / len(truths) > best_answer:
            best_t = t
            best_R = R
            best_answer = sum(truths) / len(truths)
            best_truths = all(truths)

        print(all(truths))

    if not best_truths:
        raise Exception("Something is wrong with the inputs.")

    return best_R, best_t


def fundamental_equality(F: npt.NDArray[np.float32], correspondences: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    x1, x2 = correspondences
    hx1 = np.hstack((x1, np.ones((x1.shape[0], 1)))).T
    hx2 = np.hstack((x2, np.ones((x2.shape[0], 1)))).T
    return np.diagonal(hx2.T @ F @ hx1)


def essential_equality(E: npt.NDArray[np.float32],
                       correspondences: Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]],
                       c1: float,
                       c2: float
                       ):
    # https://www.youtube.com/watch?v=zX5NeY-GTO0&list=PLgnQpQtFTOGRYjqjdZxTEQPZuFHQa7O7Y&t=2131s
    x1, x2 = correspondences
    hx1 = np.hstack((x1, np.ones((x1.shape[0], 1))*c1)).T
    hx2 = np.hstack((x2, np.ones((x2.shape[0], 1))*c2)).T
    return np.diagonal(hx2.T @ E @ hx1)


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


def isRotationMatrix(R):
    #  https://learnopencv.com/rotation-matrix-to-euler-angles/
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    #  https://learnopencv.com/rotation-matrix-to-euler-angles/
    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    import math
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ransac_iters", type=int, default=0)
    parser.add_argument("--image_stereo", type=Path, required=False)
    parser.add_argument("--image1", type=Path, default='data/towerLeft.jpg')
    parser.add_argument("--image2", type=Path, default='data/towerRight.jpg')
    parser.add_argument("--K1", type=Path, default=None)
    parser.add_argument("--K2", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.image_stereo:
        img1 = cv2.imread(str(args.image_stereo), 0)[:, :2028]  # left image
        img2 = cv2.imread(str(args.image_stereo), 0)[:, 2028:]  # right image
    elif args.image1 and args.image2:
        img1 = cv2.imread(str(args.image1), 0)
        img2 = cv2.imread(str(args.image2), 0)
    else:
        raise Exception()

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

    # plot matching points
    cv_kp1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts1]
    cv_kp2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in pts2]

    out_img = np.array([])
    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(cv_kp1))]
    out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)

    sampler = EightCorrespondenceSampler(sample_size=8)

    F, inliers = RANSAC(function=get_fundamental_matrix,
                        test_function=fundamental_equality,
                        inputs=(pts1, pts2),
                        sampler=sampler,
                        iterations=args.ransac_iters,
                        threshold=0.1)
    print(f'Fundamental matrix:\n{F}')
    print(f'Percentage of inlier points / total input points {inliers[0].shape[0]} / {pts1.shape[0]} = {inliers[0].shape[0] / pts1.shape[0]:.1%}.')

    # h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
    # h_pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
    # err = 0
    # for p1, p2 in zip(h_pts1, h_pts2):
    #     p1 = p1.reshape(-1, 1)
    #     p2T = p2.reshape(1, -1)
    #     err += np.abs(p2T @ F @ p1)

    if args.K1 and args.K2:
        K1 = np.load(args.K1)
        K2 = np.load(args.K2)
        c = K1[0, 0]
        cp = K2[0, 0]

        h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T  # Is there something wrong around here?
        h_pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1)))).T

        calibrated_pts1 = np.linalg.inv(K1) @ h_pts1
        calibrated_pts2 = np.linalg.inv(K2) @ h_pts2

        assert all(((calibrated_pts1).T)[:, 2] == 1)
        assert all(((calibrated_pts2).T)[:, 2] == 1)

        calibrated_pts1 = (calibrated_pts1.T)[:, :2]
        calibrated_pts2 = (calibrated_pts2.T)[:, :2]

        sampler = EightCorrespondenceSampler(sample_size=8)

        def get_essential_matrix_c(x, xp): return get_essential_matrix(x, xp, c, cp)
        def essential_equality_c(x, xp): return essential_equality(x, xp, c, cp)
        E, inliers = RANSAC(function=get_essential_matrix_c,
                            test_function=essential_equality_c,
                            inputs=(calibrated_pts1, calibrated_pts2),
                            sampler=sampler,
                            iterations=args.ransac_iters,
                            threshold=0.1/20)

        print(f'Essential matrix:\n{E}')
        print(f'Percentage of inlier points / total input points {inliers[0].shape[0]} / {pts1.shape[0]} = {inliers[0].shape[0] / pts1.shape[0]:.1%}.')

        # h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1))))
        # h_pts2 = np.hstack((pts2, np.ones((pts2.shape[0], 1))))
        # err = 0
        # for p1, p2 in zip(h_pts1, h_pts2):
        #     p1 = p1.reshape(-1, 1)
        #     p2T = p2.reshape(1, -1)
        #     err += np.abs(p2T @ F @ p1)

        R, t = get_baseline(E, inliers[0], inliers[1], c, cp)

        print(f'Second camera orientation:\n{R}')
        print(f'Baseline:\n{t}')

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
