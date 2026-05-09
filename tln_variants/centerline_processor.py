#!/usr/bin/env python3
"""
Post-process a raw IPS+LiDAR log into a clean centerline with track widths.

Usage:
  python3 centerline_processor.py <raw_csv> [options]

  ros2 run tln_variants centerline_processor <raw_csv> [options]

Input:  CSV produced by centerline_logger (x,y,heading,r0..r539)
Output: porto_centerline.csv  →  x_m,y_m,w_tr_right_m,w_tr_left_m
        porto_centerline.png  (with --plot)

The 540-ray LiDAR (downsampled from 1080, angle_min=-135deg, 0.5deg/ray):
  - Right 90deg wall → index 90   (-90deg from forward)
  - Left  90deg wall → index 450  (+90deg from forward)
"""

import argparse
import sys
import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# LiDAR geometry constants (540-ray downsampled scan)
_LIDAR_IDX_RIGHT = 90    # -90 deg (right wall)
_LIDAR_IDX_LEFT  = 450   # +90 deg (left wall)
_LIDAR_WIN       = 12    # half-window of rays to median over


def read_raw(path: str):
    # Infer expected column count from header to tolerate off-by-one scan lengths
    with open(path) as f:
        ncols = len(f.readline().split(','))
    data = np.loadtxt(path, delimiter=',', skiprows=1, usecols=range(ncols))
    x, y, heading = data[:, 0], data[:, 1], data[:, 2]
    lidar = data[:, 3:]
    return x, y, heading, lidar


def dedup(x, y, heading, lidar, min_dist: float = 0.05):
    """Drop consecutive points closer than min_dist metres."""
    keep = [0]
    for i in range(1, len(x)):
        d = np.hypot(x[i] - x[keep[-1]], y[i] - y[keep[-1]])
        if d >= min_dist:
            keep.append(i)
    k = np.array(keep)
    return x[k], y[k], heading[k], lidar[k]


def smooth_centerline(x, y, n_out: int, smooth: float):
    """
    Fit a closed periodic cubic B-spline through the raw points and
    resample to n_out evenly-spaced (in parameter space) points.
    """
    # Drop duplicate closing point if the driver returned to start
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 0.5:
        x, y = x[:-1], y[:-1]

    # s=smooth*N gives roughly smooth*N residual knots
    tck, _ = splprep([x, y], s=smooth * len(x), per=True, k=3)
    u_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    xs, ys = splev(u_new, tck)
    return np.asarray(xs), np.asarray(ys)


def _perp_dist(lidar_row: np.ndarray, idx: int, win: int) -> float:
    """Median LiDAR range in a window around idx, ignoring max-range hits."""
    lo = max(0, idx - win)
    hi = min(lidar_row.shape[0], idx + win + 1)
    window = lidar_row[lo:hi]
    valid = window[window < 9.5]
    if valid.size == 0:
        return float(np.min(window))  # all max-range: use raw min
    return float(np.median(valid))


def extract_widths(x_raw, y_raw, lidar_raw, x_cl, y_cl):
    """
    For each smoothed centerline point find the nearest raw sample and read
    its perpendicular LiDAR measurements as half-widths to each wall.
    """
    raw_pts = np.column_stack([x_raw, y_raw])
    n = len(x_cl)
    w_right = np.zeros(n)
    w_left  = np.zeros(n)
    for i in range(n):
        idx = int(np.argmin(np.linalg.norm(raw_pts - [x_cl[i], y_cl[i]], axis=1)))
        row = lidar_raw[idx]
        w_right[i] = _perp_dist(row, _LIDAR_IDX_RIGHT, _LIDAR_WIN)
        w_left[i]  = _perp_dist(row, _LIDAR_IDX_LEFT,  _LIDAR_WIN)
    return w_right, w_left


def main():
    ap = argparse.ArgumentParser(description='Process raw IPS log into centerline CSV')
    ap.add_argument('raw_csv', help='Path to porto_centerline_raw.csv')
    ap.add_argument('--output-dir',
                    default='/home/autodrive_devkit/src/tln_variants/tln_variants/pure_pursuit/porto',
                    help='Output directory')
    ap.add_argument('--n-points', type=int, default=500,
                    help='Resampled centerline points (default: 500)')
    ap.add_argument('--smooth', type=float, default=0.3,
                    help='Spline smoothing factor 0=interpolating (default: 0.3)')
    ap.add_argument('--width-sigma', type=float, default=5.0,
                    help='Gaussian sigma (in centerline points) for width smoothing (default: 5)')
    ap.add_argument('--max-width', type=float, default=1.5,
                    help='Hard cap on each half-width in metres (default: 1.5). '
                         'Reduces this if the raceline goes outside the track.')
    ap.add_argument('--plot', action='store_true',
                    help='Save and display verification plot')
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.raw_csv))
    os.makedirs(out_dir, exist_ok=True)

    print(f'Reading {args.raw_csv}...')
    x_raw, y_raw, heading_raw, lidar_raw = read_raw(args.raw_csv)
    print(f'  Raw samples: {len(x_raw)}')

    x_raw, y_raw, heading_raw, lidar_raw = dedup(x_raw, y_raw, heading_raw, lidar_raw)
    print(f'  After dedup: {len(x_raw)}')

    print(f'Fitting B-spline (smooth={args.smooth}, n={args.n_points})...')
    x_cl, y_cl = smooth_centerline(x_raw, y_raw, args.n_points, args.smooth)

    print('Extracting track half-widths from LiDAR...')
    w_right, w_left = extract_widths(x_raw, y_raw, lidar_raw, x_cl, y_cl)
    w_right = np.clip(w_right, 0.15, args.max_width)
    w_left  = np.clip(w_left,  0.15, args.max_width)
    print(f'  Raw widths (before smoothing): right {w_right.mean():.2f}±{w_right.std():.2f} m, '
          f'left {w_left.mean():.2f}±{w_left.std():.2f} m')

    # Smooth widths along the track to remove corner heading-misalignment artifacts.
    # The LiDAR perpendicular indices assume the vehicle is driving straight; at corners
    # the vehicle is yawing so the rays hit the walls at an angle, under-reading the width.
    # A periodic Gaussian blur irons out those dips without touching the straights.
    if args.width_sigma > 0:
        w_right = gaussian_filter1d(w_right, sigma=args.width_sigma, mode='wrap')
        w_left  = gaussian_filter1d(w_left,  sigma=args.width_sigma, mode='wrap')
        print(f'  Width smoothed (sigma={args.width_sigma} pts)')

    out_path = os.path.join(out_dir, 'porto_centerline.csv')
    np.savetxt(
        out_path,
        np.column_stack([x_cl, y_cl, w_right, w_left]),
        delimiter=',',
        header='x_m,y_m,w_tr_right_m,w_tr_left_m',
        comments='',
    )
    print(f'Centerline saved → {out_path}')
    print(f'  Width stats: right {w_right.mean():.2f}±{w_right.std():.2f} m, '
          f'left {w_left.mean():.2f}±{w_left.std():.2f} m')

    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        ax.plot(x_raw, y_raw, 'b.', ms=2, alpha=0.4, label='Raw IPS')
        ax.plot(x_cl, y_cl, 'r-', lw=2, label='Smoothed centerline')
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Porto Centerline')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        s = np.cumsum(np.hypot(np.diff(x_cl, append=x_cl[0]),
                               np.diff(y_cl, append=y_cl[0])))
        ax2.plot(s, w_right, label='Right wall dist')
        ax2.plot(s, w_left,  label='Left wall dist')
        ax2.set_xlabel('Track position [m]')
        ax2.set_ylabel('Half-width [m]')
        ax2.set_title('Track Widths')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, 'porto_centerline.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved → {plot_path}')


if __name__ == '__main__':
    main()
