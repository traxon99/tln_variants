#!/usr/bin/env python3
"""
Thin shim that runs RLNNode without timestamp encoding.
The full implementation lives in rln.py; this entry point sets use_timestamps=False.
"""
import rclpy
from tln_variants.rln import RLNNode, plot_results


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RLNNode(use_timestamps=False)
        rclpy.spin(node)
    except Exception as e:
        print(f'Fatal error: {e}')
    except KeyboardInterrupt:
        print('Node stopped by keyboard interrupt')
    finally:
        if 'node' in locals():
            plot_results(
                node.recorded_ts,
                node.recorded_speed,
                node.recorded_steer,
                filename='Figures/RLN_no_ts_speed_steering_plot.png'
            )
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
