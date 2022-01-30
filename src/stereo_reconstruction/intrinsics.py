"""
"""

from typing import List, Tuple

from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_corners(img, pattern_size=Tuple[int, int], visualize=False):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, img_corners = cv2.findChessboardCorners(img_gray, patternSize=pattern_size, corners=None)

    if ret:
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners2 = cv2.cornerSubPix(img_gray, img_corners, (11, 11), (-1, -1), criteria)

        if visualize:
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, pattern_size, corners2, ret)

            cv2.imshow('img', img)
            cv2.waitKey(200)
            # cv2.destroyAllWindows()

    else:
        print("image didn't work")

    return corners2[:, 0, :], img


def get_homography(image_coordinates, world_coordinates):
    x = image_coordinates
    X = world_coordinates

    M = np.zeros((x.shape[0]*2, 3*3))
    for idx, (xi_, Xi_) in enumerate(zip(x, X)):
        xi, yi = xi_
        Xi, Yi = Xi_
        M[idx*2] = (-Xi, -Yi, -1, 0, 0, 0, xi*Xi, xi*Yi, xi)
        M[idx*2+1] = (0, 0, 0, -Xi, -Yi, -1, yi*Xi, yi*Yi, yi)

    _, _, vh = np.linalg.svd(M)#, full_matrices=False)
    h_vec = vh[-1, :]
    h = h_vec.reshape(3, 3)

    assert np.all(np.matmul(M, h_vec) < 0.01)
    return h


def v_ij(h, i, j):
    #  https://www.youtube.com/watch?v=-9He7Nu3u8s @ 29:02
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
    #  https://www.youtube.com/watch?v=-9He7Nu3u8s @ 29:02
    return np.vstack((np.transpose(v_ij(h, 1, 2)), np.transpose(v_ij(h, 1, 1))-np.transpose(v_ij(h, 2, 2))))


def get_L(V):
    # 30:55 https://www.youtube.com/watch?v=-9He7Nu3u8s
    _, _, vh = np.linalg.svd(V, full_matrices=False)
    v = np.transpose(vh)
    b = v[:, -1]
    b11, b12, b13, b22, b23, b33 = b  # 27:37
    B = np.array(((b11, b12, b13),
                  (b12, b22, b23),
                  (b13, b23, b33)))  # 27:15
    AAT = np.linalg.cholesky(B)  # 25:29
    return AAT


def get_K(image_paths: List[Path], pattern_size: Tuple[int, int], square_size: float, visualize=False) -> np.ndarray:
    
    # Zhang's method
    # https://www.youtube.com/watch?v=-9He7Nu3u8s

    H_img = []
    V_img = []

    for path in image_paths:
        img = cv2.imread(str(path))

        x, img = get_corners(img, pattern_size, visualize)
        X = np.zeros((pattern_size[0]*pattern_size[1], 2), np.float32)
        X[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)*square_size

        h = get_homography(x, X)
        H_img.append(h)

        v = get_V(h)
        V_img.append(v)

    V = np.vstack(V_img)

    L = get_L(V)

    K = np.linalg.inv(np.transpose(L))
    K = K / K[-1,-1]

    return K


def main() -> int:
    square_size = 2.423  # 0.02423  # meters (~1")
    pattern_size = (9, 6)
    img_root = Path(r"C:\src\Tutorials\Stereo\stereo-calibration\calib_imgs\1")

    left_bad_images = [f'left{idx}' for idx in [13, 15, 16]]
    left_image_paths = [image_path for image_path in img_root.rglob("left*.jpg") if image_path.stem not in left_bad_images]
    left_K = get_K(image_paths=left_image_paths, pattern_size=pattern_size, square_size=square_size, visualize=True)

    print(left_K)

    right_bad_images = [f'right{idx}' for idx in [1, 2, 3, 4, 9, 13, 15, 16, 27, 28]]
    right_image_paths = [image_path for image_path in img_root.rglob("right*.jpg") if image_path.stem not in right_bad_images]
    right_image_paths2 = []
    right_image_paths2.extend(right_image_paths[1:8])

    right_K = get_K(image_paths=right_image_paths2, pattern_size=pattern_size, square_size=square_size, visualize=True)

    print(right_K)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
