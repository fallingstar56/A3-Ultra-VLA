# A3 Ultra ADU 抓取后停止

该组件与现有后训练模型并行运行：模型仍接收原始指令「抓瓶子」，本组件只读订阅 O10Hand 压力，满足接触条件后通过推理进程的本地 HTTP 门控锁存停止。所有新增可执行代码都在 ADU；MDU 保持厂商 MC 服务，不部署本组件。

## 接入前核实

1. 目标 ADU 的 AimDK/ROS 环境能读到 `/motion/control/hand_joint_state`，消息为 `sensor_msgs/msg/JointState`，`header.frame_id` 为 `O10Hand`。核对 QoS 为 `BEST_EFFORT`。
2. 沿用已验证的 AVATAR O10Hand 布局：`effort[-260:]` 为左 130 点、右 130 点；每手依次为五指各 16 点、掌心 25 点、手背 25 点。启动时仍检查消息类型、手型、长度和数据新鲜度。静止时全零可以是正常无接触状态。
3. 确认目标 ADU 实际运行本仓库的 `edge_infer/infer_a3_edge.py`，并且旧 AVATAR 轨迹脚本、MQTT 桥或其他程序没有同时写同一组关节。确认模型停止 chunk 后仍能稳定持瓶。
4. 监测器现使用 AVATAR 的区域分数口径：先对每区取最高 3 点均值，减去 1.5 秒空手基线的区域均值。AVATAR 的 `40` 可作本任务只读回放的初始候选；当前完成条件仍要求拇指和至少两根其他手指同时持续 150 ms、至少 3 帧。**控制阈值仍须显式填写**，并用当前动作的只读数据确认空握与持瓶可区分。
5. 只读标定开始前目标手也必须处于张开、空手且无外部接触的安全初始姿态。手已握紧时先按原机器人流程恢复姿态，不得用手掰开或继续接触采样。

## 只读采集与离线回放

在 ADU 使用系统 Python 和 ROS 环境运行订阅进程；无 `--control` 时绝不发模型启动/停止请求：

基线采集需要连续 1.5 秒数据；相邻样本间隔超过 200 ms 时重新采集。只读模式在基线完成后也会记录抓握分数，但不会 ARM 或判定任务完成。订阅仅保留最新一帧，控制请求先于日志落盘执行。日志含压力尾数组、基线、逐指分数、总分和控制事件时间。

空手全零基线现可正常完成；全零帧本身不算抓握成功。传感器断流、字段错误和超时仍会阻止动作或停止任务。

```bash
source /agibot/software/v0/entry/env/env.sh
/usr/bin/python3 edge_infer/grasp_stop/monitor.py --hand right \
  --log /agibot/data/home/agi/Desktop/grasp_stop/sample.jsonl
```

复制 JSONL 到开发电脑后，用候选阈值回放。阈值只用于回放，不写入配置：

```bash
python3 edge_infer/grasp_stop/replay.py sample.jsonl --hand right --threshold <离线候选值>
python3 -m unittest discover -s edge_infer/grasp_stop -p 'test_*.py' -v
```

## 受控试验

在已完成上述核实、现场人员监护和原有急停可用时，先启动推理进程并保持 IDLE。沿用递瓶 Demo 的 `/agibot/edge_deploy_minimal` 和 `check.sh`/`run.sh`，但必须显式启用本门控；原 Demo 的无参数 `run.sh` 会继续执行完整递送流程：

```bash
cd /agibot/edge_deploy_minimal
export A3_TASK='抓瓶子' A3_EXEC_STEPS=3
bash check.sh --grasp-stop-enabled --human_in_loop true --human_in_loop_host 127.0.0.1 --task '抓瓶子'
bash run.sh \
  --grasp-stop-enabled --human_in_loop true --human_in_loop_host 127.0.0.1 --task '抓瓶子'
```

在同一 ADU 的另一终端运行监控；该命令先采 1.5 秒空手基线，只有数据有效且阈值已显式填写时才调用本地 `/grasp/arm` 和 `/start`：

```bash
source /agibot/software/v0/entry/env/env.sh
/usr/bin/python3 edge_infer/grasp_stop/monitor.py --hand right \
  --threshold <离线标定的值> --control --confirm I_UNDERSTAND \
  --log /agibot/data/home/agi/Desktop/grasp_stop/trial.jsonl
```

推理程序的 `/start` 在未 ARM 时返回 409；完成或故障后永久锁存 IDLE，须重启推理进程才能进行新试次。监控进程每 50 ms 发本机心跳；推理进程 250 ms 未收到心跳就停止，防止监控退出后模型继续动作。抓取完成/故障时监控请求锁存，并等待取消 chunk 的回执；超时或失败会报错，现场按原安全流程接管。取消 chunk 不等于硬件急停，也不保证持续握力。

人工停止在尚未 ARM、或已 ARM 但尚未启动时也会锁存中止，随后到达的 ARM/启动请求将被拒绝；重新试验需要重启推理进程。

当前离线测试覆盖压力状态机、本机 HTTP 门控和并发 ROS worker 请求/回执配对；尚未验证新 VLA 动作中的误触发、下游已发布动作窗口的尾动、实际取消延迟和持瓶稳定性。阈值留空时不能运行 `--control`。完整真机流程见本仓库的 `真机抓瓶悬停测试流程.md`。
