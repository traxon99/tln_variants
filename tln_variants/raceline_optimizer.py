#!/usr/bin/env python3
"""
Minimum-curvature raceline optimizer for Porto (or any track).

Usage:
  python3 raceline_optimizer.py <centerline_csv> [options]
  ros2 run tln_variants raceline_optimizer <centerline_csv> [options]

Input CSV:  x_m,y_m,w_tr_right_m,w_tr_left_m  (from centerline_processor)
Output CSV: x_m,y_m,speed_mps,heading_rad      (porto_raceline.csv)
            porto_raceline.png                  (with --plot)

Vehicle (AutoDRIVE RoboRacer):
  wheelbase   0.33 m
  track width 0.236 m  (2 × 0.118 m)
  max steer   0.52 rad

Reference: Heilmeier et al., "Minimum Curvature Trajectory Planning and Control
           for an Autonomous Racecar", Vehicle System Dynamics 2019.
"""

import argparse
import os
import sys
import numpy as np
import trajectory_planning_helpers as tph

# RoboRacer geometry
W_VEH       = 0.236   # m — full vehicle width
SAFETY_M    = 0.1    # 0.05 m — extra margin per side beyond W_VEH/2
KAPPA_MAX   = 8.0     # 1/m — corresponds to ~0.125 m minimum radius

# Speed profile
V_MAX_DEFAULT   = 4.0   # m/s
A_LAT_DEFAULT   = 4.0   # m/s²
A_LONG_DEFAULT  = 2.0   # m/s² — longitudinal limit for braking/accel pass
INTERP_STEP     = 0.05  # m — raceline interpolation resolution


def _fb_pass(v: np.ndarray, el: np.ndarray, a_long: float) -> np.ndarray:
    """
    Forward-backward kinematic pass on a closed speed profile.

    Backward pass: ensures braking starts early enough before each corner.
    Forward pass:  ensures acceleration out of corners is physically reachable.

    Run each direction twice to resolve the closed-loop wrap-around.
    """
    n = len(v)
    for _ in range(2):
        # Backward — corner entry / braking
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(max(v[j] ** 2 + 2.0 * a_long * el[i], 0.0)))
    for _ in range(2):
        # Forward — corner exit / acceleration
        for i in range(n):
            j = (i + 1) % n
            v[j] = min(v[j], np.sqrt(max(v[i] ** 2 + 2.0 * a_long * el[i], 0.0)))
    return v


def load_centerline(path: str) -> np.ndarray:
    """Return Nx4 array [x, y, w_tr_right, w_tr_left], UNCLOSED."""
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    if data.shape[1] < 4:
        raise ValueError(f'Expected 4 columns, got {data.shape[1]}. '
                         'Run centerline_processor first.')
    return data[:, :4].copy()


def main():
    ap = argparse.ArgumentParser(description='Minimum-curvature raceline optimizer')
    ap.add_argument('centerline_csv', help='porto_centerline.csv from centerline_processor')
    ap.add_argument('--output-dir',
                    default='/home/autodrive_devkit/src/tln_variants/tln_variants/pure_pursuit/porto')
    ap.add_argument('--v-max',    type=float, default=V_MAX_DEFAULT, help='m/s')
    ap.add_argument('--a-lat',    type=float, default=A_LAT_DEFAULT, help='m/s² lateral limit')
    ap.add_argument('--a-long',   type=float, default=A_LONG_DEFAULT,
                    help='m/s² longitudinal limit for braking/acceleration pass. '
                         'Lower = earlier braking before corners (default: %(default)s)')
    ap.add_argument('--interp',   type=float, default=INTERP_STEP,   help='m interpolation step')
    ap.add_argument('--safety',   type=float, default=SAFETY_M,
                    help='Extra margin per side beyond vehicle half-width (m). '
                         'Increase if the raceline gets too close to walls (default: %(default)s)')
    ap.add_argument('--plot',     action='store_true')
    args = ap.parse_args()

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.centerline_csv))
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load track
    # ------------------------------------------------------------------
    print(f'Loading {args.centerline_csv}...')
    reftrack = load_centerline(args.centerline_csv)   # Nx4, UNCLOSED
    n = len(reftrack)
    print(f'  {n} centerline points')

    # Shrink allowed corridor by vehicle half-width + safety margin
    half_w = W_VEH / 2.0 + args.safety
    reftrack[:, 2] = np.maximum(reftrack[:, 2] - half_w, 0.05)
    reftrack[:, 3] = np.maximum(reftrack[:, 3] - half_w, 0.05)
    print(f'  Safety margin {args.safety:.3f} m/side  (half_w applied: {half_w:.3f} m)')

    # ------------------------------------------------------------------
    # 2. Compute splines → normvectors + system matrix M
    # ------------------------------------------------------------------
    refline_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))   # closed

    _, _, M, normvec = tph.calc_splines.calc_splines(
        path=refline_cl, use_dist_scaling=True
    )
    # normvec is Nx2 UNCLOSED, pointing left of direction of travel

    # ------------------------------------------------------------------
    # 3. Minimum-curvature QP
    # ------------------------------------------------------------------
    print('Running minimum-curvature optimization...')
    alpha, curv_err = tph.opt_min_curv.opt_min_curv(
        reftrack=reftrack,      # Nx4 UNCLOSED [x,y,w_right,w_left]
        normvectors=normvec,    # Nx2 UNCLOSED
        A=M,                    # spline system matrix
        kappa_bound=KAPPA_MAX,
        w_veh=W_VEH,
        print_debug=True,
        closed=True,
    )
    print(f'  Max curvature linearisation error: {curv_err:.4f}')

    # ------------------------------------------------------------------
    # 4. Build interpolated raceline
    # ------------------------------------------------------------------
    (
        raceline,        # Mx2 [x, y]
        _A_rl,
        _cx, _cy,
        _sp_inds, _t,
        s_rl,            # arc-length array
        _sp_lens,
        el_cl,           # closed el_lengths (M+1)
    ) = tph.create_raceline.create_raceline(
        refline=reftrack[:, :2],
        normvectors=normvec,
        alpha=alpha,
        stepsize_interp=args.interp,
    )

    # el_cl is the closed element-lengths array: N segments for N raceline points
    # (el_cl[i] = distance from point i to point (i+1) % N)
    # calc_head_curv_num expects el_lengths of the same length as path.
    el = el_cl

    # ------------------------------------------------------------------
    # 5. Heading and curvature
    # ------------------------------------------------------------------
    psi, kappa = tph.calc_head_curv_num.calc_head_curv_num(
        path=raceline,
        el_lengths=el,
        is_closed=True,
    )

    # ------------------------------------------------------------------
    # 6. Curvature-based speed profile with forward-backward kinematic pass
    #    Step 1: v = min(v_max, sqrt(a_lat / |κ|))  — lateral limit at each point
    #    Step 2: backward pass enforces braking starts early enough before corners
    #            forward pass enforces acceleration out of corners is reachable
    #    Step 3: normalise to throttle [0, 1]
    # ------------------------------------------------------------------
    v_ms = np.minimum(args.v_max,
                      np.sqrt(args.a_lat / np.maximum(np.abs(kappa), 1e-4)))
    v_ms = _fb_pass(v_ms, el, args.a_long)
    throttle = v_ms / args.v_max   # normalise: [0, 1]

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    out_path = os.path.join(out_dir, 'porto_raceline.csv')
    np.savetxt(
        out_path,
        np.column_stack([raceline, throttle, psi]),
        delimiter=',',
        header='x_m,y_m,throttle,heading_rad',
        comments='',
    )
    track_len = float(s_rl[-1])
    avg_t     = float(np.mean(throttle))
    print(f'Raceline saved → {out_path}')
    print(f'  Points   : {len(raceline)}')
    print(f'  Length   : {track_len:.1f} m')
    print(f'  Throttle : min={throttle.min():.3f}  mean={avg_t:.3f}  max={throttle.max():.3f}')

    # ------------------------------------------------------------------
    # 8. Plot
    # ------------------------------------------------------------------
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        ax = axes[0]
        ax.plot(reftrack[:, 0], reftrack[:, 1], 'b--', lw=1, alpha=0.6,
                label='Centerline')
        sc = ax.scatter(raceline[:, 0], raceline[:, 1],
                        c=throttle, cmap='viridis', s=3,
                        vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label='Throttle [0–1]')
        ax.set_aspect('equal')
        ax.set_title('Porto Raceline (color = throttle)')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.plot(s_rl, throttle, 'g-', lw=1.5)
        ax2.axhline(avg_t, color='gray', ls='--', lw=1, label=f'mean {avg_t:.3f}')
        ax2.set_xlabel('Track position [m]')
        ax2.set_ylabel('Throttle [0–1]')
        ax2.set_title('Throttle profile')
        ax2.set_ylim(0, 1.15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(out_dir, 'porto_raceline.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved → {plot_path}')


if __name__ == '__main__':
    main()
