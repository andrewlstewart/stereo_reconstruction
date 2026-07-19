"""
Dense metric stereo reconstruction.

Takes a single side-by-side stereo image (left half | right half), the saved
camera calibration (K1, K2, dist1, dist2, R, t), rectifies the pair, computes a
dense disparity map with SGBM, reprojects it to a metric 3D point cloud, colors
the cloud from the left image, and writes a colored PLY that can be opened and
rotated in any 3D viewer (MeshLab, CloudCompare, the Windows 3D Viewer, ...).

The saved extrinsics come from intrinsics.py (checkerboard homography poses):

    X_right = R @ X_left + t

with t in metres, which matches the cv2.stereoRectify convention
(cameraMatrix1 = left, cameraMatrix2 = right). Because t is metric, the
reconstruction is metric as well.

In practice you would normally reach for cv2.stereoRectify + cv2.StereoSGBM +
cv2.reprojectImageTo3D directly; this script wires those together against the
calibration produced by the rest of this repository.
"""

from typing import Optional, Tuple

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection='3d')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dense metric stereo reconstruction from a side-by-side stereo image."
    )
    parser.add_argument(
        "--calibration_path",
        type=Path,
        default=Path.cwd() / "output",
        help="Directory containing K1.npy, K2.npy, dist1.npy, dist2.npy, R.npy, t.npy.",
    )
    parser.add_argument(
        "--stereo_image",
        type=Path,
        default=Path.cwd() / "data/calibration/image01.jpg",
        help="Side-by-side stereo image (left half | right half).",
    )
    parser.add_argument(
        "--split_col",
        type=int,
        default=2028,
        help="Column separating the two views. Left = [:, :split_col], right = [:, split_col:].",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path.cwd() / "output",
        help="Directory for reconstruction.ply, rectified.jpg and disparity.jpg.",
    )
    parser.add_argument(
        "--downscale",
        type=float,
        default=1.0,
        help="Factor in (0, 1] to shrink each image before matching (speeds up SGBM).",
    )
    parser.add_argument("--min_disparity", type=int, default=0)
    parser.add_argument(
        "--num_disparities",
        type=int,
        default=0,
        help=(
            "Disparity search range (a positive multiple of 16). 0 (default) "
            "auto-sizes it from --min_depth and the rectified geometry."
        ),
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.6,
        help=(
            "Nearest expected surface in metres. Sets the disparity search "
            "range when --num_disparities is 0. Objects nearer than this cannot "
            "be matched, so keep it a little below your closest surface."
        ),
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=5,
        help="Matched block size. Odd, typically 3-11.",
    )
    parser.add_argument("--uniqueness_ratio", type=int, default=10)
    parser.add_argument("--speckle_window_size", type=int, default=100)
    parser.add_argument("--speckle_range", type=int, default=2)
    parser.add_argument("--disp12_max_diff", type=int, default=1)
    parser.add_argument(
        "--wls",
        action="store_true",
        help=(
            "Apply the ximgproc WLS disparity filter (needs opencv-contrib-python). "
            "Fills and smooths the disparity guided by the left image. Helps "
            "textured scenes; on textureless surfaces it can smear depth, so "
            "low-confidence pixels are dropped (see --wls_confidence)."
        ),
    )
    parser.add_argument("--wls_lambda", type=float, default=8000.0,
                        help="WLS smoothness (higher = smoother).")
    parser.add_argument("--wls_sigma", type=float, default=1.5,
                        help="WLS edge sensitivity to the guide image.")
    parser.add_argument(
        "--wls_confidence",
        type=float,
        default=128.0,
        help=(
            "Drop WLS pixels below this confidence (0-255) to avoid smeared "
            "depth in textureless regions. 0 keeps everything (densest, least "
            "accurate)."
        ),
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=10.0,
        help="Discard reconstructed points farther than this many metres.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="stereoRectify free scaling: 0 crops to valid pixels, 1 keeps all source pixels.",
    )
    parser.add_argument(
        "--rectified_scale",
        type=float,
        default=1.0,
        help=(
            "Widen the rectified canvas beyond the automatic un-squish (fy/fx). "
            ">1 keeps more of the scene horizontally, <1 keeps less."
        ),
    )
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Swap left/right and invert the extrinsics (use if disparities come out negative).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show an interactive, downsampled matplotlib 3D preview.",
    )
    parser.add_argument("--preview_max_points", type=int, default=60_000)
    return parser.parse_args()


def load_calibration(
    calibration_path: Path,
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load the intrinsics/extrinsics saved by intrinsics.py."""

    def _load(name: str) -> npt.NDArray[np.float64]:
        path = calibration_path / name
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration file: {path}")
        return np.load(path).astype(np.float64)

    K1 = _load("K1.npy").reshape(3, 3)
    K2 = _load("K2.npy").reshape(3, 3)
    dist1 = _load("dist1.npy").reshape(1, -1)
    dist2 = _load("dist2.npy").reshape(1, -1)
    R = _load("R.npy").reshape(3, 3)
    t = _load("t.npy").reshape(3, 1)

    return K1, dist1, K2, dist2, R, t


def split_stereo_image(
    image: npt.NDArray[np.uint8],
    split_col: int,
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Split a side-by-side capture into equally sized left/right images."""
    width = image.shape[1]
    if not 0 < split_col < width:
        raise ValueError(f"--split_col ({split_col}) must be in (0, {width}).")

    left = image[:, :split_col]
    right = image[:, split_col:]

    # Rectification and block matching require both views to be the same size.
    height = min(left.shape[0], right.shape[0])
    common_width = min(left.shape[1], right.shape[1])

    left = left[:height, :common_width]
    right = right[:height, :common_width]

    return np.ascontiguousarray(left), np.ascontiguousarray(right)


def scale_camera_matrix(
    K: npt.NDArray[np.float64],
    scale: float,
) -> npt.NDArray[np.float64]:
    """Scale intrinsics to match an image that has been resized by ``scale``."""
    K = K.copy()
    K[0, 0] *= scale  # fx
    K[1, 1] *= scale  # fy
    K[0, 2] *= scale  # cx
    K[1, 2] *= scale  # cy
    K[0, 1] *= scale  # skew scales with fx
    return K


def rectify(
    left: npt.NDArray[np.uint8],
    right: npt.NDArray[np.uint8],
    K1: npt.NDArray[np.float64],
    dist1: npt.NDArray[np.float64],
    K2: npt.NDArray[np.float64],
    dist2: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    t: npt.NDArray[np.float64],
    alpha: float,
    rectified_scale: float = 1.0,
) -> Tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    Tuple[int, int, int, int],
]:
    """Compute the rectifying transforms and warp both views into alignment.

    The stored images have non-square pixels: the two camera views are packed
    side by side, so each is squished horizontally (fy ~= 2*fx). Rectification
    produces square-pixel images, so a same-width output canvas would crop the
    horizontal field of view. The rectified canvas is therefore widened by the
    pixel aspect ratio (fy/fx) to preserve the full scene, and ``rectified_scale``
    widens it further (>1) or less (<1).
    """
    height, width = left.shape[:2]
    image_size = (width, height)

    # Un-squish: make the rectified pixels square by widening the canvas.
    aspect = float(K1[1, 1] / K1[0, 0])  # fy / fx
    new_width = int(round(width * max(aspect, 1.0) * rectified_scale))
    new_size = (new_width, height)

    R1, R2, P1, P2, Q, roi1, _roi2 = cv2.stereoRectify(
        cameraMatrix1=K1,
        distCoeffs1=dist1,
        cameraMatrix2=K2,
        distCoeffs2=dist2,
        imageSize=image_size,
        R=R,
        T=t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=alpha,
        newImageSize=new_size,
    )

    map1x, map1y = cv2.initUndistortRectifyMap(
        K1, dist1, R1, P1, new_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        K2, dist2, R2, P2, new_size, cv2.CV_32FC1
    )

    rect_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    return rect_left, rect_right, Q, roi1


def compute_disparity(
    rect_left: npt.NDArray[np.uint8],
    rect_right: npt.NDArray[np.uint8],
    args: argparse.Namespace,
) -> npt.NDArray[np.float32]:
    """Run semi-global block matching and return a float disparity map."""
    if args.num_disparities <= 0 or args.num_disparities % 16 != 0:
        raise ValueError("--num_disparities must be a positive multiple of 16.")
    if args.block_size % 2 == 0:
        raise ValueError("--block_size must be odd.")

    channels = 1
    block_size = args.block_size

    left_gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=args.min_disparity,
        numDisparities=args.num_disparities,
        blockSize=block_size,
        P1=8 * channels * block_size ** 2,
        P2=32 * channels * block_size ** 2,
        disp12MaxDiff=args.disp12_max_diff,
        uniquenessRatio=args.uniqueness_ratio,
        speckleWindowSize=args.speckle_window_size,
        speckleRange=args.speckle_range,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    if getattr(args, "wls", False):
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError(
                "--wls requires cv2.ximgproc; install opencv-contrib-python."
            )
        # Filter guided by the left image, using a right-matcher for
        # left-right consistency. Confidence masking drops the unreliable
        # (smeared) fills over textureless regions.
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        disp_left = left_matcher.compute(left_gray, right_gray)
        disp_right = right_matcher.compute(right_gray, left_gray)

        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        wls_filter.setLambda(args.wls_lambda)
        wls_filter.setSigmaColor(args.wls_sigma)
        filtered = wls_filter.filter(disp_left, rect_left, None, disp_right)

        disparity = filtered.astype(np.float32) / 16.0
        if args.wls_confidence > 0.0:
            confidence = wls_filter.getConfidenceMap()
            disparity[confidence < args.wls_confidence] = -1.0
        return disparity

    # SGBM returns a fixed-point disparity scaled by 16.
    disparity = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0

    return disparity


def build_point_cloud(
    disparity: npt.NDArray[np.float32],
    Q: npt.NDArray[np.float64],
    rect_left: npt.NDArray[np.uint8],
    roi: Tuple[int, int, int, int],
    min_disparity: int,
    max_depth: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Reproject the disparity map to a colored metric point cloud."""
    points = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    colors = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)

    # Restrict to the valid rectified region reported by stereoRectify.
    x, y, w, h = roi
    roi_mask = np.zeros(disparity.shape, dtype=bool)
    if w > 0 and h > 0:
        roi_mask[y : y + h, x : x + w] = True
    else:
        roi_mask[:] = True

    depth = points[:, :, 2]

    valid = (
        roi_mask
        & (disparity > min_disparity)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (depth <= max_depth)
    )

    return points[valid], colors[valid]


def normalize_disparity_for_display(
    disparity: npt.NDArray[np.float32],
    min_disparity: int,
    num_disparities: int,
) -> npt.NDArray[np.uint8]:
    """Map a disparity image to an 8-bit color image for inspection."""
    disp_vis = disparity.copy()
    disp_vis[disp_vis < min_disparity] = min_disparity
    disp_vis = (disp_vis - min_disparity) / float(num_disparities)
    disp_vis = np.clip(disp_vis * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)


def draw_rectified_pair(
    rect_left: npt.NDArray[np.uint8],
    rect_right: npt.NDArray[np.uint8],
    num_lines: int = 20,
) -> npt.NDArray[np.uint8]:
    """Stack the rectified pair side by side with horizontal reference lines."""
    combined = np.hstack((rect_left, rect_right))
    for i in range(1, num_lines):
        y = int(round(combined.shape[0] * i / num_lines))
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
    return combined


def write_ply(
    path: Path,
    points: npt.NDArray[np.float32],
    colors: npt.NDArray[np.uint8],
) -> None:
    """Write a colored point cloud as a binary little-endian PLY file."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)

    if points.shape[0] != colors.shape[0]:
        raise ValueError("points and colors must have the same length.")

    count = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    vertices = np.empty(
        count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["red"] = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"] = colors[:, 2]

    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())


def preview_point_cloud(
    points: npt.NDArray[np.float32],
    colors: npt.NDArray[np.uint8],
    max_points: int,
    left_image: Optional[npt.NDArray[np.uint8]] = None,
    right_image: Optional[npt.NDArray[np.uint8]] = None,
) -> None:
    """Show a downsampled, rotatable matplotlib 3D scatter of the cloud.

    When the left and right images are supplied they are drawn alongside the
    cloud so the reconstructed region can be compared with the full camera
    views. Rectification at alpha=0 crops to the valid overlap, so the cloud
    covers less of the scene than the original images.
    """
    if points.shape[0] == 0:
        print("No points to preview.")
        return

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors)

    # Robust bounds so a handful of far mismatches do not shrink the scene into
    # a tiny blob. Depth (Z) keeps a slightly longer outlier tail than X/Y, so
    # clip its far side a little more tightly -- but loosely enough to keep the
    # background (walls, doors) visible.
    lo = np.percentile(points, 2.0, axis=0)
    hi = np.percentile(points, 98.0, axis=0)
    hi[2] = np.percentile(points[:, 2], 97.0)

    inside = np.all((points >= lo) & (points <= hi), axis=1)
    if np.count_nonzero(inside) >= 100:
        points = points[inside]
        colors = colors[inside]

    if points.shape[0] > max_points:
        indices = np.random.default_rng(0).choice(
            points.shape[0], max_points, replace=False
        )
        points = points[indices]
        colors = colors[indices]

    show_images = left_image is not None and right_image is not None

    if show_images:
        figure = plt.figure(figsize=(15, 8))
        grid = figure.add_gridspec(2, 2, width_ratios=(1.0, 1.8))
        ax_left = figure.add_subplot(grid[0, 0])
        ax_right = figure.add_subplot(grid[1, 0])
        ax = figure.add_subplot(grid[:, 1], projection="3d")

        ax_left.imshow(cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB))
        ax_left.set_title("Left image")
        ax_left.axis("off")

        ax_right.imshow(cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB))
        ax_right.set_title("Right image")
        ax_right.axis("off")
    else:
        figure = plt.figure(figsize=(9, 7))
        ax = figure.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=colors / 255.0,
        s=0.5,
        marker=".",
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Equal-aspect cube centred on the robust extent: the scene fills the frame
    # and stays undistorted when rotated.
    mid = (hi + lo) / 2.0
    span = float((hi - lo).max())
    ax.set_xlim(mid[0] - span / 2.0, mid[0] + span / 2.0)
    ax.set_ylim(mid[1] - span / 2.0, mid[1] + span / 2.0)
    ax.set_zlim(mid[2] - span / 2.0, mid[2] + span / 2.0)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    # Start from the left camera's viewpoint: look straight down +Z with +X to
    # the right and +Y down, so the initial view matches the original image.
    # Drag to rotate away from this pose and reveal the scene's depth.
    ax.view_init(elev=-90.0, azim=-90.0)
    ax.set_title("Stereo reconstruction (drag to rotate)")
    plt.tight_layout()
    plt.show()


def main() -> int:
    args = parse_args()

    if not 0.0 < args.downscale <= 1.0:
        raise ValueError("--downscale must be in (0, 1].")

    if args.rectified_scale <= 0.0:
        raise ValueError("--rectified_scale must be positive.")

    K1, dist1, K2, dist2, R, t = load_calibration(args.calibration_path)

    image = cv2.imread(str(args.stereo_image))
    if image is None:
        raise FileNotFoundError(f"Could not read stereo image: {args.stereo_image}")

    left, right = split_stereo_image(image, args.split_col)

    if args.swap:
        # New left = old right, new right = old left. Invert the relative pose:
        # X_left = R.T @ X_right - R.T @ t.
        left, right = right, left
        K1, dist1, K2, dist2 = K2, dist2, K1, dist1
        R = R.T
        t = -R @ t

    if args.downscale != 1.0:
        new_size = (
            int(round(left.shape[1] * args.downscale)),
            int(round(left.shape[0] * args.downscale)),
        )
        left = cv2.resize(left, new_size, interpolation=cv2.INTER_AREA)
        right = cv2.resize(right, new_size, interpolation=cv2.INTER_AREA)
        K1 = scale_camera_matrix(K1, args.downscale)
        K2 = scale_camera_matrix(K2, args.downscale)

    rect_left, rect_right, Q, roi = rectify(
        left, right, K1, dist1, K2, dist2, R, t, args.alpha, args.rectified_scale
    )

    # Size the disparity search from the geometry so near objects are
    # resolvable. The rectified focal and baseline are read from Q:
    #   max disparity = f * baseline / min_depth.
    if args.num_disparities == 0:
        f_rect = float(Q[2, 3])
        baseline_m = 1.0 / abs(float(Q[3, 2]))
        max_disparity = f_rect * baseline_m / args.min_depth
        args.num_disparities = max(16, int(np.ceil(max_disparity / 16.0)) * 16)
        print(
            f"Auto num_disparities = {args.num_disparities} "
            f"(min_depth {args.min_depth:.2f} m, f={f_rect:.0f} px, "
            f"baseline={baseline_m * 1000:.1f} mm)"
        )

    disparity = compute_disparity(rect_left, rect_right, args)

    points, colors = build_point_cloud(
        disparity, Q, rect_left, roi, args.min_disparity, args.max_depth
    )

    baseline_mm = float(np.linalg.norm(t)) * 1000.0
    print(f"Baseline: {baseline_mm:.2f} mm")
    print(f"Reconstructed {points.shape[0]:,} points.")
    if points.shape[0] > 0:
        print(
            f"Depth range: {points[:, 2].min():.3f}-{points[:, 2].max():.3f} m "
            f"(median {np.median(points[:, 2]):.3f} m)"
        )

    args.output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(args.output_path / "rectified.jpg"),
        draw_rectified_pair(rect_left, rect_right),
    )
    cv2.imwrite(
        str(args.output_path / "disparity.jpg"),
        normalize_disparity_for_display(
            disparity, args.min_disparity, args.num_disparities
        ),
    )

    ply_path = args.output_path / "reconstruction.ply"
    write_ply(ply_path, points, colors)
    print(f"Wrote point cloud: {ply_path}")

    if args.preview:
        preview_point_cloud(
            points, colors, args.preview_max_points, left, right
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
