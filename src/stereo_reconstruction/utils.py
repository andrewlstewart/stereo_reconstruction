"""
Visualization tools for stereo imagery analysis
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def drawlines(img1, img2, lines, pts1, pts2):
    # https://github.com/softMonkeys/8-Point-Algorithm-RANSAC/blob/master/main.py
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def visualize_epipolar_lines(img1, img2, pts1, pts2, F, F_cv2=None):
      # Find epilines corresponding to points in second image,  and draw the lines on first image
    lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F).reshape(-1, 3)
    img3, img4 = drawlines(img1, img2, lines, pts1, pts2)
    nrows = 1
    if F_cv2 is not None:
        nrows = 2
    plt.close('all')
    fig, ax = plt.subplots(ncols=2, nrows=nrows)
    fig.tight_layout()
    if F_cv2 is None:
        ax[0].imshow(img3)
        ax[1].imshow(img4)
    else:
        ax[0, 0].imshow(img3)
        ax[0, 1].imshow(img4)
        lines = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F_cv2).reshape(-1, 3)
        img5, img6 = drawlines(img1, img2, lines, pts1, pts2)
        ax[1, 0].imshow(img5)
        ax[1, 1].imshow(img6)
    plt.show()
    plt.close('all')