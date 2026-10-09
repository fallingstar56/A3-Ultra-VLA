# protocol_proto/

`.proto` 源文件，vendored 自上游 `aima_protocol`（一个独立 git 仓）。

## 上游
- 路径: `/home/agiuser/Documents/e2e/aima_protocol`
- 上游 commit: `98f4f821` (2026-06-16, "Merge branch 'mono/event_trigger' into 'main'")
- Vendored 时间: 2026-06-17

## 与上游的差异

我们只 vendored 了 a3_server 真正需要的子集（**66 个 .proto**，上游有 319 个）。

差异检查（`98f4f821`）：

| 文件 | 差异 |
|---|---|
| 共同的 66 个文件中，**65 个完全一致** | ✅ |
| `protocol/ta/ta_channel.proto` | 已整份同步为上游版：含 `TaRealtimeWorkState` / `TaTakeoverState` / `TaCalibrationState` / `TaControlReferenceState` / `TaTeleopBodyMode` 5 个 enum + `TaRealtimeStatus` / `TaRealtimeStatusChannel` 2 个 message；**且不再定义 `TaWholeBodyReferenceWindow`**（该消息已迁到 `wbc_reference_window.proto`，见下）。 |

## WBC 全身链路新增（2026-07）

为 gr00t whole-body RTC 部署新增两个 `.proto`（放在 `protocol/ta/`）：

- `ta_whole_body_state.proto` → `TaWholeBodyStateChannel` / `TaWholeBodyState` / `TaJointGroupState` / `TaImuState`。
  a3_server 订阅 `/wbc/whole_body_state/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyStateChannel` 取
  leg/waist/head/arm + pelvis/torso IMU（**不含手**，手仍走 `/motion/control/hand_joint_state`）。
- `wbc_reference_window.proto` → `TaWholeBodyReferenceWindow`（从旧 `ta_channel.proto` 迁出，
  避免与 upstream ta_channel 的符号重复）。a3_server 发到
  `/wbc/infer/reference_window/pb_3Aaimdk_2Eprotocol_2ETaWholeBodyReferenceWindow`。

编出的模块名：`aimdk.protocol.ta.ta_whole_body_state_pb2`、
`aimdk.protocol.ta.wbc_reference_window_pb2`（构建脚本 `find aimdk -name '*.proto'` 自动拾取）。

也就是说：**vendored 的 .proto 在 a3_server 用得到的范围内，跟上游 `98f4f821` 字段完全等价**
（`ta_channel` / `ta_whole_body_command` 已同步到上游；`ta_whole_body_state` / `wbc_reference_window`
为上游新增消息的 vendored 拷贝）。

## 用途
- 启动时 `scripts/start_robot_a3.sh [2/3]` 调用 `protoc` 把这里所有 `.proto` 编成 pb2.py，
  输出到同级的 `_pb_gen/`（gitignored，每次启动重生成）
- ADU 端 **C++** 那边有 aima_protocol（在
  `/opt/agibot/share/ros2_package/aima_protocol_ros2_package/`，给 mc 守护进程用），
  但 **Python** 那边 `import aimdk` 是 `ModuleNotFoundError` —
  所以 a3_server 这个 Python 节点必须自己 vendor 一份 `.proto` 现场 protoc。
- 此外 `ros2_plugin_proto.msg.RosMsgWrapper` 在 ADU 上 Python 能 import
  (装在 `/opt/agibot/share/ros2_package/aimrt_protocol_ros2_package/lib/python3.12/site-packages/`)，
  这个不用 vendor。

## 协议层级（很重要）
机器人 mc 守护进程实际订阅的是 **新 layout**：

- 新 (在用): `aimdk/protocol/motion_control/motion/mc_motion_channel.proto`
  → `MotionControlMoveWaistChannel { waist_pitch=2; waist_roll=3; waist_yaw=4; waist_height=5 }` （字段扁平）
- 老 (历史保留): `aimdk/protocol/mc/motion/mc_motion_channel.proto`
  → `McMoveWaistChannel { Header header; WaistMoveValue data }` （嵌套 data）

二者 wire 不兼容。`server_node.py` 的 `_try_load_pb` loader 优先找新 layout。
保留老 layout 是为了兼容某些尚未升级的旧字段引用，将来都迁移完可以删掉
`aimdk/protocol/mc/`。

## pnc_arm 子集 (插值 / EEF 控制用)

新增 `aimdk/protocol/pnc_arm/` 4 个文件 (channel/motion/state/service)，
a3_server 用其中的 `PncArmInterpolateChannel` 发到
`/pnc_arm/motion/interpolate`，仅在 pnc_arm 切到 `PncArmControlMode_ONLINE_TRAJECTORY`
模式后生效 (用 `RoboInterface/scripts/utils/change_pnc_arm_mode_a3.sh` 切换)。

字段:
```
message PncArmInterpolateChannel {
  Header header = 1;
  uint32 flag = 2;        // 0/1/2 = joint 左/右/双; 100/101/102 = SE3 左/右/双
  repeated double positions = 3;     // 一般 17D, 前 3 维是腰占位 (实测控制不了, 填 0)
  repeated double velocities = 4;
  repeated double accelerations = 5;
  repeated double effort = 6;
}
```

## 同步上游

aima_protocol 升级后人工 sync：

```bash
# 在 aima_protocol/ 拉新版本
cd /home/agiuser/Documents/e2e/aima_protocol && git pull

# 增量同步 a3_server 用到的子集 (含 pnc_arm)
cd /home/agiuser/Documents/e2e/pi/robointerface/robotinterface/ros_server
for d in protocol/common protocol/hal protocol/mc protocol/motion_control protocol/ta protocol/pnc_arm; do
    rsync -av /home/agiuser/Documents/e2e/aima_protocol/aimdk/$d/ \
              protocol_proto/aimdk/$d/
done

# 校验差异 (除 a3_server 不用的部分外应当一致)
diff -rq /home/agiuser/Documents/e2e/aima_protocol/aimdk \
         protocol_proto/aimdk | grep -v "Only in /home/agiuser/Documents/e2e/aima_protocol"

# 启动 ADU server 验证 [2/3] protoc 步骤无报错, 测试 [4/8] WAIST 等用例通过
# 把这个 README 顶部的 commit hash 更新成新的 git rev-parse --short HEAD
```

