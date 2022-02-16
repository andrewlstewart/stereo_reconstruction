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

from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt


class HomographyIllFittedError(Exception):
    """Raised when the a.T @ homography is larger than expected."""


def get_corners(img: Tuple[str, npt.NDArray[np.float32]], pattern_size=Tuple[int, int], visualize=False):
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

        corners2 = cv2.cornerSubPix(img_gray, img_corners, (11, 11), (-1, -1), criteria)

        if visualize:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)

            print(name)
            minx, miny = np.min(corners2, axis=0)[0]
            maxx, maxy = np.max(corners2, axis=0)[0]
            plt.imshow(img[int(miny*0.9):int(maxy*1.1), int(minx*0.9):int(maxx*1.1), :])
            plt.show()

    else:
        print("image didn't work")

    return corners2[:, 0, :], img


def get_homography(image_coordinates: npt.NDArray[np.float32], world_coordinates: npt.NDArray[np.float32]):
    """
    image_coordinations and world_coordinates should be of dim (n, 2) where n is the number of corners

    Compute the homography between the model plane and its image according to:
        https://www.youtube.com/watch?v=-9He7Nu3u8s&t=10m26s

        more information on the actual computation can be found here as well
            https://www.youtube.com/watch?v=3NcQbZu6xt8&t=895s
    """

    x = image_coordinates
    X = world_coordinates

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

    # this just checks that the homography matrix is at least reasonable
    if not np.all(np.matmul(M, h_vec) < 0.01):
        raise HomographyIllFittedError()

    return h


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
                      visualize=False) -> npt.NDArray[np.float32]:
    """
    Each element in images is a tuple where the first value is the filename of the image and the second value is the numpy array

    This function follows Zhang's method as outlined in Professor Cyrill Stachniss's lecture https://www.youtube.com/watch?v=-9He7Nu3u8s
    Need a minimum of 4 points per plane and 3 views of the plane @ 31m09s
    """

    H_img = []
    V_img = []

    # Construct the checkerboard corner coordinates in the world plane of the checkerboard
    # (x, y) with z=0 for all x,y
    X = np.zeros((pattern_size[0]*pattern_size[1], 2), np.float32)
    X[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)*square_size

    # For each image compute the homography and then the V matrix
    for name, img in images:
        x, img = get_corners((name, img), pattern_size, visualize)

        h = get_homography(x, X)
        H_img.append(h)

        v = get_V(h)
        V_img.append(v)

    # For the set of images, stack them to constrcut the full V matrix
    V = np.vstack(V_img)

    B = get_B(V)
    AAT = np.linalg.cholesky(B)  # https://www.youtube.com/watch?v=-9He7Nu3u8s&t=1529s

    K = np.linalg.inv(np.transpose(AAT))
    K = K / K[-1, -1]  # TODO: Do we normalize the homogeneous coordinate if we know the scale of the square_size?

    return K


def main() -> int:
    # Constants related to the calibration pattern.
    square_size = 22.1  # mm
    pattern_size = (9, 6)
    image_root = (Path.cwd() / 'data/calibration')

    # If there are images which cause the camera matrix computation to fail, you can add them to this list and they will get ignored
    left_bad_images = []
    # My images are concatenated horizontally, so the left image is in the range 0:2027 and the right image is in the range 2028:4055
    left_images = [(image_path.name, cv2.imread(str(image_path))[:, :2028, :]) for image_path in image_root.glob('*.jpg') if image_path.stem not in left_bad_images]
    left_K = get_camera_matrix(images=left_images, pattern_size=pattern_size, square_size=square_size, visualize=False)

    print(left_K)

    right_bad_images = []
    right_images = [(image_path.name, cv2.imread(str(image_path))[:, 2028:, :]) for image_path in image_root.glob('*.jpg') if image_path.stem not in right_bad_images]

    right_K = get_camera_matrix(images=right_images[1:], pattern_size=pattern_size, square_size=square_size, visualize=False)

    print(right_K)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
