# LDRQ V3：会按指令跳舞、有人靠近就后退的宇树 R1 人形机器人

[![tests](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml/badge.svg)](https://github.com/NAPPING-AI/LDRQ/actions/workflows/tests.yml)
[![license: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

[English](README.md)

这个项目为**宇树 R1** 人形机器人做了四件事：

- 在 Isaac Lab 里训练行走策略；
- 部署到机器人自带的 Jetson Orin NX 上，不需要外接电脑；
- 在仿真和真机上用同一个变量量化 sim2real 差距；
- 加一层英文指令接口，由机器人本机运行的小语言模型解析指令。

V3 在此基础上加了一个跳舞策略，以及人靠得太近时让机器人后退的反射。

> **这里是 `v3` 分支。** 项目共三个版本，各占一个分支：`main`（v1）、`v2`、`v3`（V3，本分支）。三个版本的总览和各自的架构图见 [`main`](https://github.com/NAPPING-AI/LDRQ/tree/main) 分支的 README。

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ 封面：宇树 R1 真机与 Isaac Lab 仿真"></a>
  <br><em>点击观看 3 分钟介绍视频（<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>）：训练、部署、sim2real 测量，以及真机上的英文指令</em>
</p>

## V3（本分支）

### V3 是什么

V3 在 v2 的基础上加了三样东西，v2 的行走策略不变：
- **跳舞策略** `dance_wave`（训练记录 `2026-10-23_16-51-44_wave`）。用 BeyondMimic 的动作跟踪方法训练，原地跟随一段 10 秒的舞蹈：先右臂挥手，再左臂，然后双臂上举、屈膝两次，最后站定。双脚始终着地。
- **一个进程里跑两个策略。** 策略节点每个控制步都同时推理 v2 和跳舞策略，共用一个 CUDA 上下文。硬件桥里的状态机（*conductor*）决定哪个策略的输出送到电机。
- **近距反射**，在硬件桥里每个控制步运行一次。有人进入 0.6 米内（跳舞时 0.9 米），v2 在同一个控制步接管，后退 1.5 秒（约 0.33 米），然后站住，直到操作员确认新计划。语言模型和指令层都不在这个回路里。

对操作员来说，`ask "dance for 20 seconds"` 的行为是：
- 只有 1 米内没人、并且机器人已经站稳 1 秒，才开始跳；
- 正常结束时等到一个小节的结尾；
- 有人靠近时交给 v2 后退；
- `stop dancing` 或 Ctrl-C 会让舞蹈在下一个小节结束。

**现状。** 2026-10-24 V3 在真机上吊装跑过两轮，先输出关，再电机输出开：
- 硬件桥全程保持 RUNNING；
- 机器人能跳舞，并在小节结尾交回 v2；
- 按 Ctrl-C 后在下一个小节停下；
- 有人走到 0.79–0.88 米时，从舞蹈切换到 v2 后退；
- 有人站得太近时，拒绝开始跳舞。

还没做的：
- 脱吊装测试（R4）；
- 反射目前只对人起作用，障碍物要等用左右两目自己计算的深度；
- 输出关时出现过一次 `DEGRADED: joint_target stale`，原因还没查清。

V3 的代码全部在 [v3/](v3/README.md) 下，那里的 README 是详细说明（英文）。

### 架构

![V3 架构](docs/system_architecture_v3.svg)

一句英文指令只在机器人动之前被转换成计划：模型只负责转写，确定性代码检查计划，操作者确认后机器人才执行。计划下方还有三层，各带安全检查：指令层 10 Hz、策略 50 Hz、电机回路 500 Hz。

V3 在 v2 系统上增加的部分：
- **mission_ctl**：新增 `dance` 步骤。它以心跳的形式持续发送跳舞请求；反射让机器人后退时，结束当前计划。
- **硬件桥**：加入反射（OFF、ARMED、RETREAT、STAND 四个状态）和 conductor。每个控制步读取 `/r1_perception/proximity`，把速度指令写进 v2 的观测，再转发当前驱动策略的关节目标。
- **策略节点**：每一步构建两个策略的观测、跑两个引擎。两者合计 p99 最多 2.7 ms，预算是 4 ms。

源文件：[docs/system_architecture_v3.dot](docs/system_architecture_v3.dot)。v2 的系统图在 `v2` 分支上。[docs/system_architecture.md](docs/system_architecture.md) 讲的是 2026-09-30 的 v1 系统，指令层以下的部分相同。

**视觉模块**（[perception/](perception/README.md)，[docs/stageE_perception.md](docs/stageE_perception.md)；V3 的分叉 [v3/perception/](v3/perception/README.md)）。R1 EDU 的 Orin 上没有直接接相机，头部的双目模组通过机器人内部网络把画面推给 Orin。`r1_perception` 节点分四步工作：

1. 接收左目的 H264 视频流（UDP RTP，端口 5002，544×448，约 10 Hz），用 Jetson 的硬件解码器解码；
2. 每一帧用 YOLOX-Nano 做目标检测。引擎是在 Orin 上建的 TensorRT FP16：每帧 3.7 ms，检测结果和 FP32 一致。检测器和行走策略共用 GPU，两者同时运行时策略推理的 p99 仍在 0.6 ms 以内；
3. 同一目标要连续几帧都检测到才算确认；
4. 发布两个话题：
   - `/r1_perception/detections`（JSON）：每个物体的方位和距离，供 `turn_to`（"转向椅子"）和 `walk_to`（"走到椅子前 1.5 m 停下"）使用；另外，有人靠近或相机没有画面时，会让正在走的步骤停下；
   - `/r1_perception/proximity`：V3 的近距后退反射看的就是它。

人的距离按检测框的高度估计，这个比例不用卷尺标定：帮手先站在一臂远处，再往后退一步，各拍一次。模组自带的深度在近处不能用，它会把匹配不上的像素填成近距离。用左右两目自己计算深度的工作还在进行中。

### 开发顺序

| 阶段 | 日期（2026） | 做了什么 | 结果 |
|---|---|---|---|
| v1、v2 | 至 10-20 | 行走系统和 v2 策略，见 `main` 和 `v2` | — |
| 分叉 | 10-22 | 从 release v2.0.0 复制感知、mission_ctl、硬件桥和策略节点到 `v3/`，跑 v2 策略 | v1、v2 保持发布时的状态，可以复现 |
| 后退仿真 | 10-22 | v2 在站立、转向、行走中收到 −0.3 m/s 指令后的反应；随速度变化的停止距离 | 零摔倒；站立时后退 0.33 米 |
| 反射 | 10-22 | 参考状态机（Python）、逐拍金标、硬件桥的 C++ 移植、mission_ctl 的对接、感知的近距话题 | C++ 与参考实现逐拍一致 |
| 第一次上机 | 10-23 | 发现头部相机的深度会把近处填满：0.05–0.35 米处出现不存在的障碍物 | 改为只管人，距离按框高估计（`person_box`），免卷尺标定 |
| 舞蹈 | 10-23 | 编舞，并在 URDF 上做检查；BeyondMimic 跟踪任务；`dance_wave`（6000 次迭代，4096 个环境）；导出、生成夹具 | ONNX parity 5.3e-5（开发机） |
| 两个策略 | 10-23 | conductor（参考实现和 C++）、双引擎策略节点、仿真中的交接、`dance` 步骤和它的评测集 | 正常交接 2017 次、紧急交接 2044 次，零摔倒 |
| 真机安装 | 10-24 | `install_v3.sh` 从 v2 的目录生成 `~/r1-robot-v3`；两个引擎都在 Orin 上构建并做 parity | v2 2.3e-05，舞蹈 4.6e-05 |
| 真机，输出关 | 10-24 | 两个引擎的耗时、跳舞、有人走近、拒跳 | 两者合计每步 ≤ 2.7 ms（p99）；同一控制步交接 |
| 真机，吊装、电机输出开 | 10-24 | 跳舞、Ctrl-C、有人走近、拒跳 | 全程 RUNNING；0.79–0.88 米处后退 |
| 修复 | 10-24 至 10-26 | 跳舞步骤的回复；dry run 里的模拟桥改为 50 Hz 步进；用左右两目计算深度（开发机，进行中） | — |
| 下一步 | — | 脱吊装测试（R4）；用两目深度识别障碍物 | — |

### 重要参数

| 参数 | 取值 | 在哪里设置 |
|---|---|---|
| 行走策略 | v2（`v2_drift_ft`），不变：425 → 24，50 Hz | `models/v2_drift_ft/` |
| 跳舞策略 | 137 维输入（无历史）→ 24 个关节目标，50 Hz；动作关节、缩放、默认姿态和增益都与 v2 相同 | [v3/models/dance_wave/interface.json](v3/models/dance_wave/interface.json) |
| 舞蹈 | 10 秒一循环，共 4 个小节；只在小节边界交接 | `v3/dance/routine.py` |
| 舞蹈训练 | BeyondMimic 跟踪，Isaac Lab 2.1.0，6000 次迭代，4096 个环境，加上 v2 的随机化 | `v3/dance/r1_tracking.py`、`train_dance.py` |
| 真机 `kp_scale` | 1.3，与 v2 相同（两个策略共用一组增益） | 启动参数 |
| 反射：后退 | 连续 2 帧有人在 **0.6 米**内（障碍物 0.45 米）→ 以 (−0.3, 0, 0) m/s 后退 1.5 秒，然后站住，直到收到重置 | `v3/ros2_ws/src/r1_hw_bridge/config/bridge.yaml` 的 `reflex_*` |
| 反射：限制 | 10 秒内最多后退 2 次；再次后退前，人必须先离开到触发距离 + 0.2 米之外；画面超过 0.3 秒没更新就禁止向前走 | 同上 |
| 停止距离 | max(下限, 触发距离 + 0.10 + 1.5 · v · 延迟 + 1.3 · 制动距离(v))；下限：人 1.2 米、障碍物 0.6 米；延迟按 0.25 秒算 | 同上；`v3/reflex/stop_distance.py` |
| conductor | 开始条件：**1.0 米**内没人，且站稳 1 秒；跳舞时反射在 **0.9 米**触发；心跳 0.5 秒；最长 63 秒 | `conductor.hpp` 的默认值（`dance_*` 参数） |
| 跳舞步骤 | 5 到 60 秒，没说时长时 20 秒；没开反射时拒绝 | `v3/mission_ctl/config/mission.yaml`（`dance_*`） |
| 人的距离 | `person_box`：距离 = 0.465 / (框高 ÷ 画面高)；框同时碰到画面上下边缘 = 就在正前方 | `v3/perception/config/perception.yaml`（`person_box_k`，2026-10-24 标定） |
| mission_ctl 的反射设置 | `reflex: on`、`reflex_obstacles: off`（只管人） | `mission.yaml` |
| 启动参数 | `reflex_enable:=true dance_enable:=true dance_engine:=... dance_reference:=...`（默认都是关） | `r1_stack.launch.py` |

### 中间文件

| 文件 | 由谁生成 | 在 git 里吗 |
|---|---|---|
| `v3/dance/` 舞蹈 CSV → BeyondMimic 动作文件 `.npz`（附回放视频） | `routine.py`、`csv_to_npz_r1.py`（Isaac） | 否，可重新生成 |
| `logs/rsl_rl/r1_dance_v3/2026-10-23_16-51-44_wave/` | `v3/dance/train_dance.py` | 否（GB 级） |
| `v3/models/dance_wave/` 的 `model_5999.pt`、`policy.pt`、`policy.onnx`、`interface.json` | 训练、`export_dance.py` | 是（`SHA256SUMS`） |
| `v3/models/dance_wave/dance_ref.bin` | 策略节点播放的参考：一个循环，500 帧 | 是 |
| `v3/models/dance_wave/parity_fixture.bin` | `deploy/tools/make_fixture.py --in-dim 137` | 是 |
| `v3/reflex/golden.csv`、`golden_dance.csv` | `reflex/cases.py`、`dance_cases.py`：每个用例的每一拍，C++ 要逐拍对上 | 是 |
| 后退和交接的统计表 | `v3/sim/eval_retreat.py`、`eval_handover.py`、`summarize_retreat.py` | 数据写在 [v3/README.md](v3/README.md) |
| `r1_v3_<n>_<日期>.tar.gz` | `v3/tools/make_robot_package.sh` | 否；快速启动压缩包里带了一个 |
| `~/r1-robot-v3/`（新目录树）、`v3_models/dance_wave/policy_fp32.plan` | `install_v3.sh`，**在机器人上**，parity 通过才算装好 | 否 |
| `perception/models/yolox_fp16.plan` | `r1_build_engine --fp16`，**在机器人上**生成 | 否 |
| `~/orin_commissioning/v3*_stack.log`、`*_reflex_log.csv`（每个控制步一行）、`*_perception.log` | 机器人上的栈和感知节点 | 否 |
| 双目录制数据（`stereo_capture.py`） | 头部相机的 RTP 包，用于计算两目深度 | 否：里面可能拍到人脸 |

### 快速启动

**不需要机器人**（任何装了 Python 3.8+ 和 numpy 的机器，不需要 GPU 和 ROS）：

```bash
git clone -b v3 https://github.com/NAPPING-AI/LDRQ.git && cd LDRQ
./run_tests.sh                        # v1、v2：所有不需要 GPU、ROS 或机器人的检查
cd v3 && ./run_tests.sh               # V3：反射、conductor、C++ 与金标对比、舞蹈、mission_ctl
# 在模拟的空房间里跳舞，硬件桥的反射和 conductor 也是模拟的：结果是 DONE
python3 mission_ctl/r1_mission_cli.py run "dance 12s" --dry-run --world mission_ctl/tests/worlds/room.json
# 同样跳舞，但有人走近：v2 接管并后退
python3 mission_ctl/r1_mission_cli.py run "dance 12s" --dry-run --world mission_ctl/tests/worlds/approach.json
```
`v3/run_tests.sh` 里的 C++ 检查需要支持 C++17 的 g++。硬件桥本身也能在开发机上对着假机器人编译（见 [v3/README.md](v3/README.md#on-a-dev-box)）。

**在机器人上，用 release：** [v3.0.0](https://github.com/NAPPING-AI/LDRQ/releases/tag/v3.0.0)（预发布）把所有内容（包括语言模型）放在两个压缩包里，用一个 `install.sh` 安装，安装方式与 2026-10-24 真机上相同。使用说明见 [v3/release/README_zh.md](v3/release/README_zh.md)（English：[README.md](v3/release/README.md)），包括安装、每次上机、指令、验收检查、标定、故障排查和回滚。`v3/release/make_release_v3.sh` 可以从已提交的代码重新生成它。

**在机器人上，用本分支的快速启动包：** [quickstart/ldrq-v3-quickstart.tar.gz](quickstart/ldrq-v3-quickstart.tar.gz)（约 17 MB）分两部分：
- `base/`：v2 的机器人目录树，包括硬件桥和策略节点的源码、`policy_pack`、`mission_ctl`、模型服务的脚本、v2 和 v1 的策略包，以及带 YOLOX-Nano 和 parity 夹具的视觉节点；
- `v3_package/`：由 `v3/tools/make_robot_package.sh` 打包的 V3，带跳舞策略，包括 `install_v3.sh`、按机器人目录结构排好的 V3 代码，以及安装程序要校验的哈希清单。

这正是 2026-10-24 真机上走过的路径：先在一个目录里装好 v2，再由 `install_v3.sh` 从它生成新目录 `~/r1-robot-v3`，v2 的目录保持原样。这个压缩包已在开发机上用 `install_v3.sh --rehearsal` 演练过：8 步全部通过，源目录没被改动，新目录里 V3 的测试全部通过。

语言模型的运行时和权重约 1.8 GB，放不进 git，指南里写了从哪里获取。[quickstart/make_quickstart.sh](quickstart/make_quickstart.sh) 可以从已提交的代码重新生成这个压缩包。

### 上机指南

[quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md)（English：[quickstart/ROBOT_GUIDE.md](quickstart/ROBOT_GUIDE.md)）包括：
1. 安装 v2 底座、构建视觉引擎、安装 V3；
2. 相机设置和免卷尺标定；
3. 输出关的检查；
4. 吊装、电机输出开的检查。

每一步都写了要检查什么、什么情况下必须停下。

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
| V3 舞蹈 + 近距后退（仿真） | 跳舞中交还给 v2：正常结束 2017 次、紧急交接 2044 次，**零摔倒** |
| V3 真机（吊装，电机输出开） | 按指令跳舞、在小节结尾结束；有人走到 **0.9 m** 内立即交给 v2 后退；1 m 内有人时拒绝起跳；两个策略每拍合计 **≤ 2.7 ms**（p99） |

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
| [v3/README.md](v3/README.md) | V3：舞蹈策略、两个策略的交接、近距后退反射、真机结果（英文） |
| [quickstart/ROBOT_GUIDE_zh.md](quickstart/ROBOT_GUIDE_zh.md) | V3 上机指南 |
| [docs/system_architecture.md](docs/system_architecture.md) | LLM 与运控的完整数据流、架构图和逻辑图 |
| [docs/stageA_kp_sweep.md](docs/stageA_kp_sweep.md) | 核心实验：`kp_scale` 稳定域，仿真与真机画在同一张图上 |
| [docs/project_history.md](docs/project_history.md) | 项目过程：周计划、验收门槛（PG/FR 编号的含义），以及每周的发现 |

许可证：Apache-2.0。R1 的 URDF 和网格文件来自宇树，采用 BSD-3-Clause，详见 [NOTICE](NOTICE)。
