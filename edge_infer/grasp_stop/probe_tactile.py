#!/usr/bin/env python3
"""Read-only O10Hand probe comparing rest and one labeled contact."""

import statistics
import sys
import time

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
    def capture(seconds):
        samples.clear()
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        return list(samples)

    try:
        input("双手不接触物体，按回车采空手基线：")
        baseline_rows = capture(2.0)
        input(f"保持轻触【{label}】，按回车后继续保持 2 秒：")
        contact_rows = capture(2.0)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not baseline_rows or not contact_rows:
        raise SystemExit("空手或接触阶段未收到 O10Hand 消息")
    n = min(map(len, baseline_rows + contact_rows))
    baseline = [
        statistics.median(row[i] for row in baseline_rows)
        for i in range(n)
    ]
    contact = [
        statistics.median(row[i] for row in contact_rows)
        for i in range(n)
    ]
    changed = [
        (i, round(contact[i] - baseline[i], 1))
        for i in range(n)
        if abs(contact[i] - baseline[i]) >= 1
    ]
    changed.sort(key=lambda pair: abs(pair[1]), reverse=True)
    print(
        f"label={label} baseline_frames={len(baseline_rows)} "
        f"contact_frames={len(contact_rows)} effort_len={n}"
    )
    print(f"changed_count={len(changed)}")
    print(f"largest_changes={changed[:80]}")


if __name__ == "__main__":
    main()
