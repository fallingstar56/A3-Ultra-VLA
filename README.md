# GR00T_deploy

A3 ADU whole-body 端侧推理的最小代码仓库。模型权重、Cosmos、ONNX、
TensorRT engine、导出目录和运行记录不进入 Git。

仓库保留实际推理链路、ROS server 依赖、ONNX 导出与 TensorRT 编译代码，
并按部署要求保留 AArch64 ADU 使用的 `torch_venv` 和 `trt_venv`。

## 目录

```text
gr00t_deploy/
├── edge_infer/                 # 推理入口、RTC client、TRT policy、ROS worker
├── RoboInterface/              # interface/config/token chunk 协议
├── gr00t_code/
│   ├── gr00t/                  # GR00T 运行模块
│   └── scripts/deployment/     # ONNX 导出、TRT 编译与验证工具
├── ros_server/
│   ├── _pb_gen/                # 生成的 protobuf Python 代码
│   ├── python_deps/            # 与 pb2 匹配的 protobuf runtime
│   ├── install/                # a3_server、joint_msgs、ros2_plugin_proto
│   ├── src/                    # 对应 ROS package 源码
│   └── protocol_proto/         # protobuf 源定义
├── torch_venv/                 # AArch64/Thor PyTorch 推理环境
├── trt_venv/                   # AArch64 TensorRT/ONNX 工具环境
├── env.sh                      # 公共环境变量
├── prepare_checkpoint.sh       # 初始化新 checkpoint 与 deploy.env
├── check.sh                    # 完整预检，不发送动作
├── run.sh                      # 启动推理
├── export_onnx.sh              # 导出到 work/export
├── build_engines.sh            # 从 ONNX 编译到 work/build
└── convert_model.sh            # 串行执行 ONNX 导出与 engine 编译
```

## 拉取代码

虚拟环境中的两个大文件使用 Git LFS 存储。首次拉取时执行：

```bash
git lfs install
git clone https://code.agibot.com/embodied_infra/gr00t_deploy.git
cd gr00t_deploy
git lfs pull
```

两个虚拟环境面向机器人 ADU 的 AArch64/Python 3.12 系统镜像，不是在普通
x86_64 工作站上运行的通用环境。

## 准备模型

Git 不包含以下目录和产物：

- `models/`
- `Cosmos-Reason2-2B/`
- `*.safetensors`、`*.onnx`、`*.onnx.data`、`*.engine`
- `work/`、`chunks/`、日志和缓存

部署时把相应 checkpoint 与 Cosmos 放到部署目录，或通过软链接引用已有副本，这里以172.23.20.240和172.23.20.41两台机器为例：

```text
/agibot/edge_deploy_minimal/
├── models/
│   ├── checkpoint-50000/       # 172.23.20.240 使用
│   └── checkpoint-60000/       # 172.23.20.41 使用
└── Cosmos-Reason2-2B/
```

模型配置中的 Cosmos 路径必须指向目标机器上的实际目录。

### 初始化一个新 checkpoint

从训练目录新复制 checkpoint 后，先运行初始化脚本。它会：

- 校验 checkpoint shard、processor/statistics 和 Cosmos 必需文件；
- 从训练配置自动解析 embodiment、hand/gripper 和训练 prompt；
- 生成 `<checkpoint>/deploy.env`；
- 备份并修正 `config.json`、`processor_config.json` 中的 Cosmos 路径；
- 创建 `onnx/`、`engines/`、`chunks/` 目录。

```bash
cd /agibot/edge_deploy_minimal

bash prepare_checkpoint.sh \
  --checkpoint /agibot/edge_deploy_minimal/models/checkpoint-60000 \
  --cosmos /agibot/edge_deploy_minimal/Cosmos-Reason2-2B
```

初始化脚本不会猜测缺失的 prompt 或 hand kind。checkpoint 配置不完整或保存了
多个 prompt 时，它会停止并要求部署人员先核对训练记录。

后续 `export_onnx.sh` 和 `build_engines.sh` 会自动使用 `deploy.env` 中解析出的
`EMBODIMENT`。只有旧 checkpoint 无法解析时，才应在核对训练配置后显式设置：

```bash
export A3_EMBODIMENT_TAG=NEW_EMBODIMENT
```

## 选择 checkpoint

推荐显式设置绝对路径，如：

```bash
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-50000
```

也可以只指定目录名；`env.sh` 会在仓库的 `models/` 下查找：

```bash
export A3_CHECKPOINT_NAME=checkpoint-60000
```

若两者都没有设置，默认使用 `models/checkpoint-50000`。设置
`A3_MODEL_PATH` 时，它的优先级高于 `A3_CHECKPOINT_NAME`。

## 两台机器运行推理脚本示例命令

### 172.23.20.240：checkpoint-50000

先登录 HDU，再进入内部 ADU `10.42.10.11`：

```bash
ssh agi@172.23.20.240
ssh agi@10.42.10.11

cd /agibot/edge_deploy_minimal
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-50000
export A3_EXEC_STEPS=3

bash check.sh
sleep 5
bash run.sh
```

### 172.23.20.41：checkpoint-60000

先登录 HDU，再进入该机器内部的 ADU `10.42.10.11`：

```bash
ssh agi@172.23.20.41
ssh agi@10.42.10.11

cd /agibot/edge_deploy_minimal
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000
export A3_EXEC_STEPS=15

bash check.sh
sleep 5
bash run.sh
```

`check.sh` 会加载 checkpoint 和 TensorRT engine、启动传感器 worker，但不会
安装或发送动作 chunk。预检退出后等待 5 秒，是为了让 ROS/DDS 中的临时
`/a3_server` 节点完成清理，再启动正式推理。

初始化日志全部输出完后，最后会显示：

```text
[ws] 请输入 s 启动推理；输入 p 暂停推理
```

输入 `s` 开始推理，输入 `p` 暂停推理。

## 运动控制模式

开始 whole-body 推理前，在 MDU 上进入 whole-body tracking：

```bash
python3 /agibot/software/v0/config/motion_control/tools/mc_action_cli.py \
  AVATAR --avatar-mode whole-body-tracking
```

推理暂停并退出后，在 MDU 上切回 `MOTION`：

```bash
python3 /agibot/software/v0/config/motion_control/tools/mc_action_cli.py MOTION
```

## ONNX 导出

导出完整 pipeline，默认写入 `work/export`，示例：

```bash
cd /agibot/edge_deploy_minimal
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-50000
bash export_onnx.sh
```

导出 另一个ckpt 时只需改模型路径，如：

```bash
A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000 \
  bash export_onnx.sh
```

## TensorRT engine 编译

`build_engines.sh` 的输入优先级为：

1. 显式设置的 `A3_ONNX_SOURCE_DIR`；
2. `export_onnx.sh` 刚生成的 `work/export/onnx`；
3. 已验证 checkpoint 中保留的 `${A3_MODEL_PATH}/onnx`。

编译结果写入 `work/build/engines`：

```bash
cd /agibot/edge_deploy_minimal
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-50000
bash build_engines.sh
```

也可以指定其他 ONNX 目录：

```bash
A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000 \
A3_ONNX_SOURCE_DIR=/path/to/onnx \
  bash build_engines.sh
```

两个脚本都接受额外的 `build_trt_pipeline.py` 参数。engine 与 GPU 架构、
TensorRT 版本及静态 shape 绑定；当前环境面向 Thor `sm_101` 和
TensorRT 10.13。

### 一条命令完成模型转换

对于新 checkpoint，推荐直接运行：

```bash
cd /agibot/edge_deploy_minimal
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000
bash convert_model.sh
```

该命令严格串行执行：

```text
checkpoint -> work/export/onnx -> work/build/engines
```

成功后会生成 `work/conversion.env`。使用新编译的 engine 做预检或推理前执行：

```bash
source /agibot/edge_deploy_minimal/work/conversion.env
bash check.sh
sleep 5
bash run.sh
```

完整转换会额外使用约 12–15 GB 临时空间，执行前应先检查 `df -h /agibot`。

同一目录连续转换多个 checkpoint 时，建议给每个模型指定独立 work 目录，避免旧的
ONNX symlink 或 engine 输出与另一个模型混用：

```bash
export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000
export A3_EXPORT_OUTPUT_DIR=/agibot/edge_deploy_minimal/work/checkpoint-60000/export
export A3_BUILD_OUTPUT_DIR=/agibot/edge_deploy_minimal/work/checkpoint-60000/build
bash convert_model.sh
source /agibot/edge_deploy_minimal/work/conversion.env
```

新 checkpoint 从初始化到无动作预检的完整顺序如下：

```bash
cd /agibot/edge_deploy_minimal

bash prepare_checkpoint.sh \
  --checkpoint /agibot/edge_deploy_minimal/models/checkpoint-60000 \
  --cosmos /agibot/edge_deploy_minimal/Cosmos-Reason2-2B

export A3_MODEL_PATH=/agibot/edge_deploy_minimal/models/checkpoint-60000
bash convert_model.sh

source /agibot/edge_deploy_minimal/work/conversion.env
bash check.sh
```

输出位置：

```text
work/export/onnx/       新导出的 ONNX（包含 dit_bf16.onnx.data）
work/build/engines/     新编译的 7 个 TensorRT engine
work/conversion.env     让 check.sh/run.sh 使用新 engine 的环境变量
```

`convert_model.sh` 不会覆盖 checkpoint 内原来保留的 `onnx/` 和 `engines/`。完成准确性
验收后，再由部署人员决定是否将 work 中的产物提升为新的基线。

## 系统依赖

仓库自带主要 Python 依赖，但仍要求目标 ADU 系统提供：

- CUDA 驱动、cuBLAS、cuDNN
- TensorRT 10.13 动态库
- ROS 2 Jazzy、`rclpy`、`cv_bridge` 和常用 ROS message
- `/agibot/software/v0` 中的 AimRT/机器人协议运行时
- Sonic、WBC、MC 等机器人端服务

不要在 Thor 上直接用通用 `pip install torch tensorrt` 覆盖仓库内版本。

## Git 内容边界

`.gitignore` 明确排除了模型和所有推理/导出产物；`torch_venv` 与
`trt_venv` 则被明确重新包含，不会因为其中存在 ONNX 包测试数据、缓存命名
或二进制文件而被误排除。两个超过 100 MiB 的 PyTorch 动态库由 Git LFS
管理。

推理链路生成的 `work/`、`chunks/` 和日志可在本地按需清理，它们不会进入
Git 历史。
