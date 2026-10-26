# LDRQ v2：能前后左右走、听懂英文指令的宇树 R1 人形机器人

[![tests](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml/badge.svg)](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[English](README.md)

这个项目为**宇树 R1** 人形机器人做了四件事：

- 在 Isaac Lab 里训练行走策略；
- 部署到机器人自带的 Jetson Orin NX 上，不需要外接电脑；
- 在仿真和真机上用同一个变量量化 sim2real 差距；
- 加一层英文指令接口，由机器人本机运行的小语言模型解析指令。

> **这里是 `v2` 分支。** 项目共三个版本，各占一个分支：`main`（v1）、`v2`（本分支）、`v3`（V3）。三个版本的总览和各自的架构图见 [`main`](https://github.com/NAPPING-AI/LDRQ/tree/main) 分支的 README。

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ 封面：宇树 R1 真机与 Isaac Lab 仿真"></a>
  <br><em>点击观看 3 分钟介绍视频（<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>）：训练、部署、sim2real 测量，以及真机上的英文指令</em>
</p>

## v2（本分支）

### v2 是什么

v2 在 v1 的基础上重训了行走策略 `v2_drift_ft`（训练记录 `2026-10-16_17-57-14_v2_drift_ft`）：
- 能**后退**（最快 0.6 m/s）和**横移**（±0.5 m/s），原地转向最快 1.57 rad/s，是 v1 的三倍；
- 零指令时原地踏步，不再往前蹭：仿真中每 10 秒 0.14 米，v1 是 0.45 米；
- 接口（425 维输入 → 24 个关节目标，50 Hz）和增益都与 v1 相同，所以硬件桥的 C++ 代码、关节映射和安装流程都不用改。

策略之外的变化：
- 指令层能走四个方向（`walk backward 1m`、`walk left 3s`）。速度范围取自 v2 策略包，安装时写进硬件桥的配置；
- 语言模型改到 CPU 上运行（4 个线程），GPU 留给控制回路；
- stage E 的视觉节点（`perception/`，在头部相机画面上跑 YOLOX-Nano）已经在代码里，供 `turn_to` 和 `walk_to` 两个步骤使用。它的引擎、以及与策略共用 GPU 的测试已经在真机上通过；这两个步骤本身还没在真机上跑过，所以快速启动包里和 release 一样，`perception` 是 `off`。

v2 发布为 [v2.0.0](https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0)，2026-10-20 起在真机上使用。本分支就是这个版本。完整说明见 [docs/v2_policy.md](docs/v2_policy.md)。

### 架构

![v2 架构](docs/system_architecture_v2.svg)

一句英文指令只在机器人动之前被转换成计划：模型只负责转写；确定性代码对照策略的训练范围检查计划；操作者确认后机器人才执行。计划下方还有三层，各带安全检查：指令层 10 Hz、策略 50 Hz、电机回路 500 Hz。

相对 v1 的变化：
- 多了一个策略文件，硬件桥的速度范围更宽；
- 模型改到 CPU：检测器和策略共用 GPU 时，策略推理的 p99 仍在 0.6 ms 以内；
- 视觉节点：通过机器人内部网络接收头部相机的左目画面（UDP RTP、H264、端口 5002），用 Jetson 的硬件解码器解码，再用 TensorRT FP16 引擎跑 YOLOX-Nano（每帧 3.7 ms）。

源文件：[docs/system_architecture_v2.dot](docs/system_architecture_v2.dot)。[docs/system_architecture.md](docs/system_architecture.md) 讲的是 2026-09-30 的 v1 系统，指令层以下的部分和 v2 相同；视觉节点见 [docs/stageE_perception.md](docs/stageE_perception.md)。

### 开发顺序

| 阶段 | 日期（2026） | 做了什么 | 结果 |
|---|---|---|---|
| v1（W01–W08，stage A–D） | 至 10-02 | v1 系统和 release v1.1.0，见 `main` | — |
| stage E，开发机 | 至 10-14 | 视觉节点：YOLOX-Nano、跟踪器、`turn_to` / `walk_to`、相机守卫；英文提示词加入这两个步骤 | 开发机上测试和仿真试验通过 |
| stage E，真机 | 10-14 | 在 Orin 上构建引擎、做 parity；模型改到 CPU；策略、检测器、模型三者共存测试 | FP16 引擎通过；全部运行时策略 p99 < 0.6 ms |
| v2，诊断 | 10-16 | 测出 v1 零指令往前蹭是策略本身的偏差（仿真中 +0.046 m/s） | 原因：零在训练范围的边缘 |
| v2，训练 | 10-16 | 跑了四次；保留的那次先训 4000 次迭代形成步态，再加漂移惩罚训 800 次 | 每 10 秒蹭 0.14 米；5 个 seed × 256 个环境零摔倒 |
| v2，打包 | 10-16 至 10-20 | 策略包；指令层支持四个方向；视觉节点改为读取头部相机的视频流（`source: stereo`） | 开发机：策略包校验通过，ONNX 与 PyTorch 差 1.5e-5 |
| v2，真机 | 10-20 | 用 `policy_pack` 安装，在 Orin 上重建引擎；发布 v2.0.0 | 能后退、横移，零指令时站定 |

### 重要参数

加粗的是相对 v1 有变化的，其余与 v1 相同（见 `main`）。

| 参数 | 取值 | 在哪里设置 |
|---|---|---|
| 接口 | 425 → 24，50 Hz，与 v1 相同 | `models/v2_drift_ft/interface/` |
| PD 增益、`kp_scale` | 与 v1 相同（腿 100 / 2 ……）；真机 **1.3** | `interface/actuator_gains.json`、启动参数 |
| **速度范围** | **vx −0.6 → 1.0 m/s，vy ±0.5 m/s，wz ±1.57 rad/s** | [tasks/r1_flat/flat_v2_env_cfg.py](tasks/r1_flat/flat_v2_env_cfg.py) 的 `COMMAND_RANGES_FINAL_V2`；安装时写进 `bridge.yaml` |
| **零指令环境比例** | **15 %**（v1 是 2 %） | `flat_v2_env_cfg.py` |
| **速度跟踪奖励** | v1 的宽核（std 0.5），**加一个窄核，std 0.15，权重 0.5** | 同上 |
| **漂移惩罚** | **零指令时约 1 秒内的平均平面速度，权重 −5，只在第 4000 次迭代之后加** | 同上；`tasks/r1_flat/mdp` 里的 `zero_command_drift` |
| **两脚间距惩罚** | **两脚横向距离小于 0.07 米时惩罚** | 同上 |
| **训练** | **4000 + 800 次迭代**（checkpoint `model_4799.pt`），4096 个环境，seed 42 | `flat_v2_env_cfg.py` 的文档字符串 |
| **指令层** | **后退默认 0.3 m/s，横移默认 0.2 m/s**；前进 0.4 m/s、转向 0.4 rad/s | `mission_ctl/config/mission.yaml` |
| **模型位置** | **CPU，4 线程**（`llm_num_gpu: 0`、`llm_num_thread: 4`） | 同上 |
| 视觉（stage E） | YOLOX-Nano，TensorRT **FP16**；每 4 个策略步一帧（12.5 Hz）；置信度 0.30；连续 3 帧确认；1.2 米内有人或 0.6 米内有障碍物时停止行走 | `perception/config/perception.yaml` |
| 相机 | 头部双目的左目，UDP RTP H264，端口 5002，544 × 448；深度来自 `/dev/video-dep`（先执行 `sudo unitree_depth_start`），`walk_to` 需要 | 同上（`source: stereo`） |

### 中间文件

| 文件 | 由谁生成 | 在 git 里吗 |
|---|---|---|
| `logs/rsl_rl/r1_flat/2026-10-16_15-51-33_v2/`（第一阶段）、`..._17-57-14_v2_drift_ft/`（第二阶段） | `scripts/train_r1.py` | 否（GB 级）；保留的策略在 `models/v2_drift_ft/` |
| `models/v2_drift_ft/model_4799.pt`、`policy.pt`、`policy.onnx`、`parity_fixture.bin` | 训练、`scripts/play_r1.py` 导出、夹具工具 | 是（`SHA256SUMS`） |
| `models/v2_drift_ft/interface/`（接口、增益） | `deploy/tools/dump_interface.py --run`、`dump_actuator_gains.py` | 是 |
| 指令网格表（漂移、后退、横移、转向） | `scripts/eval_command_grid.py` | 数据写在 [docs/v2_policy.md](docs/v2_policy.md) |
| 策略包 `bundles/<run_id>/`（v2，以及用于回滚的 v1） | `policy_pack/make_bundle.py --task-cfg tasks/r1_flat/flat_v2_env_cfg.py ...` | 否；打包进快速启动压缩包 |
| `deploy/artifacts/policy_fp32.plan`、`installed.json` | `install_bundle.sh`，**在机器人上**，parity 通过后生成 | 否 |
| `bridge.yaml` 里的速度范围 | `install_bundle.sh` 第 4 步，取自策略包的 `command_envelope.json` | 否：仓库里提交的仍是 v1 的范围，由安装程序改写 |
| `perception/models/yolox_nano.onnx`、`dog.jpg` | `perception/models/fetch_yolox.sh`（按 sha256 固定的下载） | 否；在快速启动压缩包里 |
| `perception/models/yolox_fixture.bin` | `perception/tools/make_fixture.py`（用 YOLOX 自带的样例图，和 stage E 一样） | 否；在快速启动压缩包里 |
| `perception/models/yolox_fp16.plan` | `r1_build_engine --fp16`，**在机器人上**生成 | 否 |
| stage E 的结果（引擎、共存、仿真试验、英文评测） | stage E 的工具 | 整理在 [docs/stageE_perception.md](docs/stageE_perception.md) |

### 快速启动

**不需要机器人**（任何装了 Python 3.8+ 的机器，不需要 GPU 和 ROS）：

```bash
git clone -b v2 https://github.com/NAPPING-AI/LDRQ.git && cd LDRQ
./run_tests.sh                                    # 不需要 GPU、ROS 或机器人
# 用 v2 策略的速度范围仿真一个计划：后退和横移
printf '{"vx": [-0.6, 1.0], "vy": [-0.5, 0.5], "wz": [-1.57, 1.57]}' > /tmp/v2_env.json
python3 mission_ctl/r1_mission_cli.py run "walk backward 1m; walk left 2s; turn right 180@1.0" --dry-run --envelope /tmp/v2_env.json
# 在模拟房间里走到椅子前 1.2 米
python3 mission_ctl/r1_mission_cli.py run "walk_to the chair stop_at 1.2; turn left 90" --dry-run --world mission_ctl/tests/worlds/room.json
```

**在机器人上：** [quickstart/ldrq-v2-quickstart.tar.gz](quickstart/ldrq-v2-quickstart.tar.gz)（约 9 MB）包含 v2 在机器人上运行所需的全部内容：
- 硬件桥和策略节点的源码，在机器人上编译；
- `policy_pack`、`mission_ctl`、模型服务的脚本；
- v2 策略包，以及用于回滚的 v1 策略包；
- 视觉节点，带 YOLOX-Nano 和它的 parity 夹具；
- 自检脚本。

语言模型的运行时和权重约 1.8 GB，放不进 git，指南里写了从哪里获取。[quickstart/make_quickstart.sh](quickstart/make_quickstart.sh) 可以从已提交的代码重新生成这个压缩包。

### 上机指南

[quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md)（English：[quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md)）包括：
- 从把压缩包拷到机器人上，到用英文指令后退、横移的全过程；
- 从 v1 升级、回滚到 v1；
- 可选的视觉检查。

想做训练或仿真：见英文 README 的 [*Reproducing*](README.md#reproducing) 一节。

## 结果一览

| 项目 | 结果 |
|---|---|
| 仿真速度跟踪（0.5–1.0 m/s） | 最大误差 **0.037 m/s**（门槛 0.15），零摔倒 |
| 真机 TensorRT 输出与 PyTorch 对比 | 512 组输入最大差 **1.3e-5**（门槛 1e-3） |
| Orin 上的策略推理（50 Hz 控制回路内） | **约 0.45 ms**，控制周期 20 ms |
| 不挂吊装连续行走 | **≥ 79.7 秒** |
| 能容忍的执行器刚度倍数 `kp_scale` | 仿真 **0.80–2.00+**，真机 **1.10–1.50**；差距来自躯干姿态，不是关节跟踪 |
| 真机闭环转向 | 10/10 次，按 IMU 读数误差都在 **1.6°** 以内 |
| 英文句子 → 经检查的计划（留出测试集） | Orin 上 **77/80**，中位耗时 0.65 秒；端到端 11 条指令全部正确 |
| 换策略 | **一条命令**、8 步检查；3 个坏包全部在改动任何文件之前被拒 |
| v2 策略：零指令（仿真） | 往前蹭的距离从 **每 10 秒 0.45 m 降到 0.14 m**（v1 → v2）；能后退到 0.6 m/s、横移 ±0.5 m/s、转向到 1.4 rad/s，零摔倒 |
| v2 真机 | 能**后退、侧行**，零指令时站定原地踏步；用同一条命令安装 |

## 亮点

- **仿真到真机的完整链路，全部在机器人本机运行。**
  - 训练：PPO + 非对称 actor-critic，五项域随机化，5 帧观测历史。
  - 部署：策略导出为 ONNX，在 Orin 上构建 TensorRT 引擎，用 parity 校验后，由 C++/ROS 2 硬件桥驱动。
- **用一张图量化 sim2real 差距。**
  - 仿真和真机跑同一串指令、用同一个稳定性判据，各自扫描关节刚度。
  - 结果：真机能容忍的范围最多只有仿真预测的三分之一；原因是躯干姿态，不是关节跟踪。
- **大部分真实缺陷都是"静默"的：** 链路报告成功，却给出看似合理的错误结果。报告里逐条列出了它们，例如：
  - TensorRT 的 `--fp32` 其实启用了 TF32；
  - 没切开发者模式时，厂家运控会和我们同时写电机指令，表现很像 sim2real 差距；
  - 奖励项 `feet_air_time_positive_biped` 的最优解是单腿站着不迈步。
- **"模型只转写、代码来判定"的指令层。**
  - 每个距离、时间、角度都必须在句子里明确说出来；
  - 超出训练范围的指令会被拒绝，并说明原因；
  - 回报内容由模板生成，不由模型生成。
- **可复现。**
  - 部署用的策略随仓库提供，无需重训；
  - CI 在每次推送时运行全部无 GPU 的检查；
  - 每个分支带一个快速启动压缩包和中英文上机指南；端侧 [release](https://github.com/kedianwen/LDRQ/releases) 附中文安装说明。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/technical_report.md](docs/technical_report.md) | 技术报告：整个项目的完整叙述 |
| [docs/v2_policy.md](docs/v2_policy.md) | v2 策略：v1 零指令往前蹭的原因、两阶段重训、后退与横移、真机部署 |
| [docs/stageE_perception.md](docs/stageE_perception.md) | stage E 视觉：YOLOX-Nano、`turn_to` / `walk_to`、真机上的引擎与共存测试 |
| [quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md) | v2 上机指南 |
| [docs/system_architecture.md](docs/system_architecture.md) | LLM 与运控的完整数据流、架构图和逻辑图 |
| [docs/stageA_kp_sweep.md](docs/stageA_kp_sweep.md) | 核心实验：`kp_scale` 稳定域，仿真与真机画在同一张图上 |
| [docs/project_history.md](docs/project_history.md) | 项目过程：周计划、验收门槛（PG/FR 编号的含义），以及每周的发现 |

许可证：Apache-2.0。R1 的 URDF 和网格文件来自宇树，采用 BSD-3-Clause，详见 [NOTICE](NOTICE)。
