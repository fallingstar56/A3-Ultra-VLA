# A3 Ultra ADU 抓取后停止

该组件与现有后训练模型并行运行：模型仍接收原始指令「抓瓶子」，本组件只读订阅 O10Hand 压力，满足接触条件后通过推理进程的本地 HTTP 门控锁存停止。所有新增可执行代码都在 ADU；MDU 保持厂商 MC 服务，不部署本组件。

## 接入前核实

1. 目标 ADU 的 AimDK/ROS 环境能读到 `/motion/control/hand_joint_state`，消息为 `sensor_msgs/msg/JointState`，`header.frame_id` 为 `O10Hand`。核对 QoS 为 `BEST_EFFORT`。
2. 用只读采样确认 `effort[-260:]` 确为压力：左 130 点、右 130 点；每手依次为五指各 16 点、掌心 25 点、手背 25 点。此布局来自 AVATAR 的触觉实验脚本，官方文档没有明确字段，所以必须对目标手型/固件逐区触碰验证。全零数据、左右响应颠倒或缺字段都禁止启用。
3. 确认目标 ADU 实际运行本仓库的 `edge_infer/infer_a3_edge.py`，并且旧 AVATAR 轨迹脚本、MQTT 桥或其他程序没有同时写同一组关节。确认模型停止 chunk 后仍能稳定持瓶。
4. 对空手、空握、触碰未抓稳、稳定持瓶等数据离线回放标定 `T_GRASP`。**生产阈值故意没有默认值**；AVATAR 旧脚本里的 `40` 不适用本任务。

## 只读采集与离线回放

在 ADU 使用系统 Python 和 ROS 环境运行订阅进程；无 `--control` 时绝不发模型启动/停止请求：

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

在已完成上述核实、现场人员监护和原有急停可用时，先启动推理进程并保持 IDLE。现有启动脚本可把附加参数传入推理程序：

```bash
A3_TASK='抓瓶子' bash edge_infer/run_a3_adu_wholebody_rtc.sh \
  --grasp-stop-enabled --human_in_loop true --human_in_loop_host 127.0.0.1
```

在同一 ADU 的另一终端运行监控；该命令先采 1 秒空手基线，只有数据有效且阈值已显式填写时才调用本地 `/grasp/arm` 和 `/start`：

```bash
source /agibot/software/v0/entry/env/env.sh
/usr/bin/python3 edge_infer/grasp_stop/monitor.py --hand right \
  --threshold <离线标定的值> --control --confirm I_UNDERSTAND \
  --log /agibot/data/home/agi/Desktop/grasp_stop/trial.jsonl
```

推理程序的 `/start` 在未 ARM 时返回 409；完成或故障后永久锁存 IDLE，须重启推理进程才能进行新试次。监控进程每 50 ms 发本机心跳；推理进程 250 ms 未收到心跳就停止，防止监控退出后模型继续动作。抓取完成/故障时监控请求锁存，并等待取消 chunk 的回执；超时或失败会报错，现场按原安全流程接管。取消 chunk 不等于硬件急停，也不保证持续握力。

目前代码只做离线和本机 HTTP 测试，尚未在目标机器人上验证 ROS 压力字段、实际取消延迟和持瓶稳定性。阈值留空时不能运行 `--control`。
