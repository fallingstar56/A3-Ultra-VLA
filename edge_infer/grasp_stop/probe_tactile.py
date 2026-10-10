#!/usr/bin/env python3
"""Read-only O10Hand pressure probe for one labeled contact condition."""

import statistics
import sys

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from sensor_msgs.msg import JointState


def main():
    if len(sys.argv) != 2:
        raise SystemExit("用法: probe_tactile.py 空手|右拇指|右食指|...")

    label = sys.argv[1]
    samples = []
    rclpy.init()
    node = rclpy.create_node("a3_labeled_tactile_probe")
    qos = QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )

    def on_msg(msg):
        if msg.header.frame_id == "O10Hand":
            samples.append(tuple(msg.effort))

    subscription = node.create_subscription(
        JointState,
        "/motion/control/hand_joint_state",
        on_msg,
        qos,
    )
    try:
        for _ in range(250):
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not samples:
        raise SystemExit("未收到 O10Hand 消息")
    n = min(map(len, samples))
    values = [
        statistics.median(sample[i] for sample in samples)
        for i in range(n)
    ]
    start = max(0, n - 260)
    tail_nonzero = [
        (i, values[i])
        for i in range(start, n)
        if values[i] != 0
    ]
    print(f"label={label} frames={len(samples)} effort_len={n}")
    print(f"tail_nonzero={tail_nonzero[:80]}")
    print(f"tail_nonzero_count={len(tail_nonzero)}")


if __name__ == "__main__":
    main()
