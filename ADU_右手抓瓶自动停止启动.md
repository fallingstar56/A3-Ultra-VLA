# ADU 拉取代码后：右手抓瓶自动停止

本流程使用 `edge_infer/grasp_stop/calibration.py` 的 `right_bottle_v1`：右拇指加食指最低区域分数达到 20，连续 1.5 秒后锁存停止；任务超时 180 秒。只使用右手。日志保存在 `/agibot/grasp_stop/`。本流程的 `--control` 命令会自动调用 `/grasp/arm` 和 `/start`，机器人随即开始运动。VLA 的任务文本沿用 checkpoint 的 `deploy.env`；若没有该设置，则使用初始版启动器的默认文本 `A3U hands over the water bottle`。压力配置名称与模型任务文本无关。

## 1. MDU：WBT 模式

确认 MOTION 模式的手部控制程序已退出，再在 MDU 执行：

```bash
python3 /agibot/software/v0/config/motion_control/tools/mc_action_cli.py \
  AVATAR --avatar-mode whole-body-tracking
python3 /agibot/software/v0/config/motion_control/tools/mc_action_cli.py --list
```

确认当前为 `AVATAR`、`WHOLE_BODY_TRACKING`，右手没有握持或接触物体，没有其他程序向手或 WBC 发送目标。保留原先成功推理时的右手初始姿态；原先是闭手，就不要额外张手。现场人员准备接管、托瓶和急停。不要在 WBT 模式运行 MOTION 开手脚本。

## 2. ADU 终端 A：环境、版本和无动作预检

每次运行使用新的编号。下列三个 ADU 终端中的 `GRASP_TRIAL_DIR` 必须相同。

```bash
source /agibot/software/v0/entry/env/env.sh
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /agibot/edge_deploy_minimal11
export A3_MODEL_PATH=/agibot/edge_deploy_minimal11/models/checkpoint-50000
unset A3_TASK  # 使用原先成功推理时的 checkpoint/default 任务文本
export A3_EXEC_STEPS=3
export GRASP_TRIAL_DIR=/agibot/grasp_stop/auto_right_001
export A3_RECORD_CHUNKS_DIR="$GRASP_TRIAL_DIR/chunks"
set -o pipefail

test -f edge_infer/grasp_stop/calibration.py
test -f ros_server/install/setup.bash
test -d ros_server/_pb_gen
test -d "$A3_MODEL_PATH/engines"
mkdir -p "$GRASP_TRIAL_DIR"
grep -Fn 'requires --task 抓瓶子' edge_infer/infer_a3_edge.py || true
grep -F 'class HeartbeatSender' edge_infer/grasp_stop/monitor.py
date -Ins | tee "$GRASP_TRIAL_DIR/start_time.txt"
sha256sum edge_infer/grasp_stop/{calibration,detector,monitor}.py \
  | tee "$GRASP_TRIAL_DIR/code_sha256.txt"

/usr/bin/python3 - <<'PY'
import sys
sys.path.insert(0, 'edge_infer/grasp_stop')
from calibration import RIGHT_BOTTLE_V1
print(RIGHT_BOTTLE_V1)
assert (RIGHT_BOTTLE_V1.threshold, RIGHT_BOTTLE_V1.contact_mode,
        RIGHT_BOTTLE_V1.dwell_seconds, RIGHT_BOTTLE_V1.task_timeout_seconds) == \
       (20.0, 'thumb_index', 1.5, 180.0)
PY

bash check.sh \
  --grasp-stop-enabled \
  --human_in_loop true \
  --human_in_loop_host 127.0.0.1 \
  2>&1 | tee "$GRASP_TRIAL_DIR/preflight.log"

sleep 5
```

预检应显示 `sensor check passed` 和 `preflight passed; no action chunk was sent`。核对 `[a3-rtc] task:` 与原先成功推理时完全一致；若不一致，先核对 checkpoint 的 `deploy.env` 和原实验命令。失败时停止在此处；预检不会发送动作 chunk。

## 3. ADU 终端 B：启动推理，保持 IDLE

```bash
source /agibot/software/v0/entry/env/env.sh
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /agibot/edge_deploy_minimal11
export A3_MODEL_PATH=/agibot/edge_deploy_minimal11/models/checkpoint-50000
unset A3_TASK  # 与预检保持一致
export A3_EXEC_STEPS=3
export GRASP_TRIAL_DIR=/agibot/grasp_stop/auto_right_001
export A3_RECORD_CHUNKS_DIR="$GRASP_TRIAL_DIR/chunks"
set -o pipefail

bash run.sh \
  --grasp-stop-enabled \
  --human_in_loop true \
  --human_in_loop_host 127.0.0.1 \
  2>&1 | tee "$GRASP_TRIAL_DIR/infer.log"
```

等初始化完成。不要在这个终端按 `s`；不要调用 HTTP `/start`。此时应保持 `IDLE`、`UNARMED`。

## 4. ADU 终端 C：确认状态并启动压力控制

```bash
source /agibot/software/v0/entry/env/env.sh
export ROS_LOCALHOST_ONLY=0
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
cd /agibot/edge_deploy_minimal11
export GRASP_TRIAL_DIR=/agibot/grasp_stop/auto_right_001
set -o pipefail

curl --noproxy '*' -fsS http://127.0.0.1:5100/status \
  | tee "$GRASP_TRIAL_DIR/status_before.json"
```

确认 JSON 中 `state` 为 `IDLE`、`grasp_state` 为 `UNARMED`。确认右手没有接触瓶子，保持原先成功推理时的姿态（闭手也可以），接着执行下面的命令。它先采集 1.5 秒无接触的触觉基线，然后**自动启动 VLA**：

```bash
/usr/bin/python3 edge_infer/grasp_stop/monitor.py \
  --hand right \
  --profile right_bottle_v1 \
  --control --confirm I_UNDERSTAND \
  --log "$GRASP_TRIAL_DIR/pressure.jsonl" \
  2>&1 | tee "$GRASP_TRIAL_DIR/monitor.log"
```

看到 `[grasp-stop] ARMED and model started` 表示正在运行。看到 `contact confirmed; stop latched` 表示压力条件已触发，程序会等待 `cancel_chunk` 回执并退出。模型也可能因传感器故障或 180 秒超时而停止。

若看到 `monitor heartbeat expired; cancelling chunk`，说明服务端超过 0.25 秒没有收到监控心跳，已将状态锁存为 `SENSOR_UNAVAILABLE` 并取消 chunk。检查 `/status` 中 `cancel_ack_epoch >= stop_epoch` 且 `cancel_failed=false`；该进程不能再次 `/start`，下一次运行需重启终端 B 的推理进程并使用新的试验目录。监控器的心跳由独立线程发送；若压力处理主循环停顿超过 0.2 秒，心跳线程也会停止，以保留失活保护。

## 5. 状态核对、人工中止与收尾

在 ADU 的另一终端可以随时人工停止；现场急停仍按原机器人流程操作：

```bash
export GRASP_TRIAL_DIR=/agibot/grasp_stop/auto_right_001
curl --noproxy '*' -fsS -X POST http://127.0.0.1:5100/stop \
  | tee "$GRASP_TRIAL_DIR/manual_stop.json"
```

自动停止或人工中止后，检查最终状态：

```bash
export GRASP_TRIAL_DIR=/agibot/grasp_stop/auto_right_001
curl --noproxy '*' -fsS http://127.0.0.1:5100/status \
  | tee "$GRASP_TRIAL_DIR/status_after.json"
```

自动抓握停止应看到 `state=IDLE`、`grasp_state=GRASP_CONFIRMED`、`cancel_ack_epoch >= stop_epoch`。机器人停止推理后可能失去持续握力，现场人员继续托住瓶子并观察。然后在终端 B 按 `Ctrl+C` 退出推理进程；确认 ADU 不再发送 VLA 指令后，在 MDU 按原流程切回 `MOTION`：

```bash
python3 /agibot/software/v0/config/motion_control/tools/mc_action_cli.py MOTION
```

一次停止即锁存。下一次运行使用新的 `/agibot/grasp_stop/auto_right_XXX` 目录，并重启推理和监测进程。
