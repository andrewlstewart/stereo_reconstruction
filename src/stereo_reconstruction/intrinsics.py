"""
An implementation of Zhang's method toc ompute the camera calibration matrix from a set of images of a flat checkerboard
    Flexible Camera Calibration By Viewing a Plane From Unknown Orientations
    Zhengyou Zhang
    http://www.vision.caltech.edu/bouguetj/calib_doc/papers/zhan99.pdf

Much of this implementation follows the course given by Professor Cyrill Stachniss's and his lectures:
    https://www.youtube.com/watch?v=-9He7Nu3u8s
    and
    https://www.youtube.com/watch?v=3NcQbZu6xt8

No effort has been made to account for distortions.

This code is mostly just an exercise to understand some deeper computer vision concepts.

In practice, you should probably just call cv2.calibrateCamera https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
"""

from typing import Sequence, Tuple

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt


class TupleArgSplitter(argparse.Action):
    """
    Splits command line arguments into tuples.

    i.e., --option 9,5 becomes namespace.option = (9,5)
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        values_ = tuple([int(val) for val in values.split(',')])
        setattr(namespace, self.dest, values_)


class HomographyIllFittedError(Exception):
    """Raised when the a.T @ homography is larger than expected."""


def get_corners(img: Tuple[str, npt.NDArray[np.float32]], pattern_size:Tuple[int, int], visualize=False):
    """
    Standard implementation to get corners from a checkerboard pattern
    https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    """
    name, img = img

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size, corners=None)

    if ret:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        corners = cv2.cornerSubPix(img_gray, img_corners, (11, 11), (-1, -1), criteria)

        if visualize:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)

            print(name)
            minx, miny = np.min(corners, axis=0)
            maxx, maxy = np.max(corners, axis=0)
            plt.imshow(img[int(miny*0.9):int(maxy*1.1), int(minx*0.9):int(maxx*1.1), :])
            plt.show()

    else:
        print("image didn't work")

    return corners, img


def homography_reprojection_errors(
    H: npt.ArrayLike,
    world_coordinates: npt.ArrayLike,
    image_coordinates: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    H = np.asarray(H, dtype=np.float64)
    world_coordinates = np.asarray(
        world_coordinates,
        dtype=np.float64,
    ).reshape(-1, 2)

    image_coordinates = np.asarray(
        image_coordinates,
        dtype=np.float64,
    ).reshape(-1, 2)

    world_h = np.column_stack(
        [
            world_coordinates,
            np.ones(world_coordinates.shape[0]),
        ]
    )

    projected_h = (H @ world_h.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]

    return np.linalg.norm(projected - image_coordinates, axis=1)


def hartley_normalization(x):
    # Described here: Hartley R. In Defence of the 8-point algorithm.
    # https://users.cecs.anu.edu.au/~hartley/Papers/fundamental/fundamental.pdf
    # 5.1 Isotropic Scaling

    u_bar, v_bar = np.mean(x, axis=0)
    d = np.mean(np.sqrt(np.pow(x[:,0] - u_bar, 2) + np.pow(x[:,1] - v_bar, 2)))
    s = np.sqrt(2) / d
    T = np.array(((s, 0, -s*u_bar), (0, s, -s*v_bar), (0, 0, 1)))
    x_h = np.hstack((x,np.ones(x.shape[0])[:, None]))
    x_n = np.matmul(T, x_h.T).T
    x_n = x_n[:, :2] / x_n[:, 2:]
    return x_n, T


def get_homography(image_coordinates: npt.NDArray[np.float32], world_coordinates: npt.NDArray[np.float32]):
    """
    image_coordinations and world_coordinates should be of dim (n, 2) where n is the number of corners

    Compute the homography between the model plane and its image according to:
        https://www.youtube.com/watch?v=-9He7Nu3u8s&t=10m26s

        more information on the actual computation can be found here as well
            https://www.youtube.com/watch?v=3NcQbZu6xt8&t=895s
    """

    x, x_T = hartley_normalization(image_coordinates)
    X, X_T = hartley_normalization(world_coordinates)

    M = np.zeros((x.shape[0]*2, 3*3))
    for idx, (xi_, Xi_) in enumerate(zip(x, X)):
        xi, yi = xi_
        Xi, Yi = Xi_
        M[idx*2] = (-Xi, -Yi, -1, 0, 0, 0, xi*Xi, xi*Yi, xi)
        M[idx*2+1] = (0, 0, 0, -Xi, -Yi, -1, yi*Xi, yi*Yi, yi)

    _, _, vh = np.linalg.svd(M)
    # vh is returned transposed, we want the last column of the non-transposed which is the same as the last row of the transposed
    h_vec = vh[-1, :]
    h = h_vec.reshape(3, 3)

    H = np.linalg.inv(x_T) @ h @ X_T

    # Homographies are defined only up to a nonzero scale.
    if abs(H[2, 2]) > np.finfo(np.float64).eps:
        H /= H[2, 2]
    else:
        H /= np.linalg.norm(H)

    errors = homography_reprojection_errors(
        H,
        world_coordinates,
        image_coordinates,
    )

    # print(f"Mean reprojection error: {np.mean(errors):.4f} px")
    # print(f"Median reprojection error: {np.median(errors):.4f} px")
    # print(f"Maximum reprojection error: {np.max(errors):.4f} px")

    return H


def v_ij(h: npt.NDArray[np.float32], i: int, j: int):
    """
    Compute the individual v_ij values according to https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1742s

    This is really inelegant but many slide decks have ambiguous notation or incorrect notation.
    This strategy made it clear as to the correct notation.
    """
    ((h11, h12, h13), (h21, h22, h23), (h31, h32, h33)) = h
    h = {'11': h11, '21': h21, '31': h31,
         '12': h12, '22': h22, '32': h32,
         '13': h13, '23': h23, '33': h33}
    return np.array((h[f'1{i}']*h[f'1{j}'],
                     h[f'1{i}']*h[f'2{j}'] + h[f'2{i}']*h[f'1{j}'],
                     h[f'3{i}']*h[f'1{j}'] + h[f'1{i}']*h[f'3{j}'],
                     h[f'2{i}']*h[f'2{j}'],
                     h[f'3{i}']*h[f'2{j}'] + h[f'2{i}']*h[f'3{j}'],
                     h[f'3{i}']*h[f'3{j}']
                     )).reshape(6, 1)


def get_V(h):
    """
    Compute the V matrix values according to https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1742s
    """
    return np.vstack((np.transpose(v_ij(h, 1, 2)), np.transpose(v_ij(h, 1, 1))-np.transpose(v_ij(h, 2, 2))))


def get_B(V: npt.NDArray[np.float32]):
    """
    Compute B from the decomposition of V
    https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1855s
    """
    _, _, vh = np.linalg.svd(V, full_matrices=False)
    b = vh[-1, :]
    b11, b12, b13, b22, b23, b33 = b  # https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1657s
    B = np.array(((b11, b12, b13),
                  (b12, b22, b23),
                  (b13, b23, b33)))
    return B


def get_camera_matrix(images: Sequence[Tuple[str, npt.NDArray[np.float32]]],
                      pattern_size: Tuple[int, int],
                      square_size: float,
                      visualize=False,
                      opencv_vals=(False, ())) -> npt.NDArray[np.float64]:
    """
    Each element in images is a tuple where the first value is the filename of the image and the second value is the numpy array

    This function follows Zhang's method as outlined in Professor Cyrill Stachniss's lecture https://www.youtube.com/watch?v=-9He7Nu3u8s
    Need a minimum of 4 points per plane and 3 views of the plane @ 31m09s
    """
    use_OpenCV, flags = opencv_vals
    if use_OpenCV:
        objpoints = []
        imgpoints = []   
    else:
        H_img = []
        V_img = []

    # Construct the checkerboard corner coordinates in the world plane of the checkerboard
    # (x, y) with z=0 for all x,y
    X = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    X[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)*square_size

    # For each image compute the homography and then the V matrix
    for name, img in images:
        x, img = get_corners((name, img), pattern_size, visualize)

        if use_OpenCV:
            objpoints.append(X)
            imgpoints.append(x)
        else:
            h = get_homography(x, X[:, :2])
            H_img.append(h)

            v = get_V(h)
            V_img.append(v)

    if use_OpenCV:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1][1:], None, None, flags=flags)
    else:
        # For the set of images, stack them to construct the full V matrix
        V = np.vstack(V_img)

        B = get_B(V)

        B = 0.5 * (B + B.T)

        eigenvalues = np.linalg.eigvalsh(B)

        if np.all(eigenvalues < 0):
            B = -B
            eigenvalues = np.linalg.eigvalsh(B)

        if not np.all(eigenvalues > 0):
            raise ValueError(
                f"B is not positive definite: eigenvalues={eigenvalues}"
            )

        AAT = np.linalg.cholesky(B)  # https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1529s

        K = np.linalg.inv(np.transpose(AAT))
        K = K / K[-1, -1]  
        # is required because B, and therefore K, is recovered only up to projective scale. It is unrelated to the physical checkerboard square size.

        dist = np.zeros((1,5))

    return K, dist


def pose_from_homography(
    H: npt.ArrayLike,
    K: npt.ArrayLike,
    world_xy: npt.ArrayLike,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Recover the board-to-camera pose from a planar homography.

    H maps checkerboard coordinates [X, Y, 1] into image coordinates.

    Because world_xy is expressed in physical units, t is recovered
    in those same units.
    """
    H = np.asarray(
        H,
        dtype=np.float64,
    ).reshape(3, 3)

    K = np.asarray(
        K,
        dtype=np.float64,
    ).reshape(3, 3)

    world_xy = np.asarray(
        world_xy,
        dtype=np.float64,
    ).reshape(-1, 2)

    # A = K^-1 H = scale * [r1, r2, t]
    A = np.linalg.solve(K, H)

    a1 = A[:, 0]
    a2 = A[:, 1]
    a3 = A[:, 2]

    # In an ideal homography, ||a1|| and ||a2|| are equal.
    scale = 2.0 / (
        np.linalg.norm(a1)
        + np.linalg.norm(a2)
    )

    def construct_pose(
        pose_scale: float,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        r1 = pose_scale * a1
        r2 = pose_scale * a2
        t = pose_scale * a3

        r3 = np.cross(r1, r2)

        R_approx = np.column_stack(
            (r1, r2, r3)
        )

        # Project the approximate matrix onto SO(3).
        U, _, Vt = np.linalg.svd(R_approx)

        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(U @ Vt)

        R = U @ correction @ Vt

        return R, t

    R, t = construct_pose(scale)

    # Resolve the projective sign by requiring the checkerboard
    # to lie in front of the camera.
    world_xyz = np.column_stack(
        (
            world_xy,
            np.zeros(world_xy.shape[0]),
        )
    )

    camera_points = (
        R @ world_xyz.T
    ).T + t

    if np.median(camera_points[:, 2]) < 0.0:
        R, t = construct_pose(-scale)

    return R, t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stereo_images_path", type=Path, default=Path.cwd() / 'data/calibration')
    parser.add_argument("--square_size", type=float, default='0.0221')  # m
    parser.add_argument("--pattern_size", action=TupleArgSplitter, default=(9,6))
    parser.add_argument("--output_path", type=Path, default=Path.cwd() / 'output')
    parser.add_argument("--use_opencv", action="store_true")
    parser.add_argument("--opencv_comparison", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Constants related to the calibration pattern.
    square_size = args.square_size
    pattern_size = args.pattern_size
    image_root = args.stereo_images_path

    image_paths = sorted(image_root.glob("*.jpg"))

    flags = 0
    if args.opencv_comparison:
        flags = (
            cv2.CALIB_ZERO_TANGENT_DIST
            | cv2.CALIB_FIX_K1
            | cv2.CALIB_FIX_K2
            | cv2.CALIB_FIX_K3
            | cv2.CALIB_FIX_K4
            | cv2.CALIB_FIX_K5
            | cv2.CALIB_FIX_K6
        )

    # # If there are images which cause the camera matrix computation to fail, you can add them to this list and they will get ignored
    if args.use_opencv:
        left_bad_images = []
        right_bad_images = []
    else:
        left_bad_images = []
        right_bad_images = ['image01', 'image02', 'image03', 'image04', 'image05', 'image06']

    # My images are concatenated horizontally, so the left image is in the range 0:2027 and the right image is in the range 2028:4055
    left_images = [
        (path.name, cv2.imread(str(path))[:, :2028, :])
        for path in image_paths
        if path.stem not in left_bad_images
    ]

    right_images = [
        (path.name, cv2.imread(str(path))[:, 2028:, :])
        for path in image_paths
        if path.stem not in right_bad_images
    ]

    left_K, left_distortion = get_camera_matrix(images=left_images, pattern_size=pattern_size, square_size=square_size, visualize=False, opencv_vals=(args.use_opencv, flags))

    print(left_K)

    args.output_path.mkdir(parents=True, exist_ok=True)
    np.save(args.output_path / 'K1.npy', left_K)
    np.save(args.output_path / 'dist1.npy', left_distortion)

    right_K, right_distortion = get_camera_matrix(images=right_images, pattern_size=pattern_size, square_size=square_size, visualize=False, opencv_vals=(args.use_opencv, flags))

    print(right_K)

    np.save(args.output_path / 'K2.npy', right_K)
    np.save(args.output_path / 'dist2.npy', right_distortion)

    ###

    world_xyz = np.zeros(
        (
            pattern_size[0] * pattern_size[1],
            3,
        ),
        dtype=np.float64,
    )

    world_xyz[:, :2] = (
        np.mgrid[
            0:pattern_size[0],
            0:pattern_size[1],
        ]
        .T
        .reshape(-1, 2)
        * square_size
    )

    world_xy = world_xyz[:, :2]

    left_by_name = dict(left_images)
    right_by_name = dict(right_images)

    common_names = sorted(
        set(left_by_name) & set(right_by_name)
    )

    baseline_estimates = []
    translation_estimates = []
    rotation_estimates = []

    for name in common_names:
        corners1, _ = get_corners(
            (f"{name} left", left_by_name[name]),
            pattern_size,
        )

        corners2, _ = get_corners(
            (f"{name} right", right_by_name[name]),
            pattern_size,
        )

        corners1 = corners1.reshape(-1, 2)
        corners2 = corners2.reshape(-1, 2)

        corners1 = cv2.undistortPoints(
            corners1.reshape(-1, 1, 2),
            left_K,
            left_distortion,
            P=left_K,
        ).reshape(-1, 2)

        corners2 = cv2.undistortPoints(
            corners2.reshape(-1, 1, 2),
            right_K,
            right_distortion,
            P=right_K,
        ).reshape(-1, 2)

        H1 = get_homography(corners1, world_xy)
        H2 = get_homography(corners2, world_xy)

        R1, t1 = pose_from_homography(
            H1,
            left_K,
            world_xy,
        )

        R2, t2 = pose_from_homography(
            H2,
            right_K,
            world_xy,
        )

        R21 = R2 @ R1.T
        t21 = t2 - R21 @ t1

        baseline_i = np.linalg.norm(t21)

        print(
            f"{name}: baseline={baseline_i * 1000:.3f} mm, "
            f"t={t21}"
        )

        rotation_estimates.append(R21)
        translation_estimates.append(t21)
        baseline_estimates.append(baseline_i)

    baseline_estimates = np.asarray(baseline_estimates)

    print(
        "Median baseline: "
        f"{np.median(baseline_estimates) * 1000:.3f} mm"
    )

    print(
        "Baseline range: "
        f"{np.min(baseline_estimates) * 1000:.3f}–"
        f"{np.max(baseline_estimates) * 1000:.3f} mm"
    )

    rotation_estimates = np.asarray(
        rotation_estimates,
        dtype=np.float64,
    )

    translation_estimates = np.asarray(
        translation_estimates,
        dtype=np.float64,
    )

    baseline_estimates = np.linalg.norm(
        translation_estimates,
        axis=1,
    )

    q1, q3 = np.percentile(
        baseline_estimates,
        [25.0, 75.0],
    )

    iqr = q3 - q1

    inlier_mask = (
        (baseline_estimates >= q1 - 1.5 * iqr)
        & (baseline_estimates <= q3 + 1.5 * iqr)
    )

    accepted_rotations = rotation_estimates[inlier_mask]
    accepted_translations = translation_estimates[inlier_mask]

    # Robust metric translation.
    t_robust = np.median(
        accepted_translations,
        axis=0,
    )

    # Average rotations, then project the result back onto SO(3).
    R_average = np.mean(
        accepted_rotations,
        axis=0,
    )

    U, _, Vt = np.linalg.svd(
        R_average,
        full_matrices=False,
    )

    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(U @ Vt)

    R_robust = U @ correction @ Vt

    print(
        "Robust stereo rotation:",
        R_robust,
        sep="\n",
    )

    print(
        "Robust metric translation:",
        t_robust,
    )

    print(
        "Robust baseline: "
        f"{np.linalg.norm(t_robust) * 1000.0:.3f} mm"
    )

    np.save(
        args.output_path / "R.npy",
        R_robust,
    )

    np.save(
        args.output_path / "t.npy",
        t_robust,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
