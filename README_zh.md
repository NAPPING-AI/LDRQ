# LDRQ：让宇树 R1 人形机器人从仿真走到真机，并听懂英文指令

[English](README.md)

这个项目为**宇树 R1** 人形机器人做了四件事：

- 在 Isaac Lab 里训练行走策略；
- 部署到机器人自带的 Jetson Orin NX 上，不需要外接电脑；
- 在仿真和真机上用同一个变量量化 sim2real 差距；
- 加一层英文指令接口，由机器人本机运行的小语言模型解析指令。

<p align="center">
  <a href="docs/media/ldrq_pitch.mp4"><img src="docs/media/cover.jpg" width="720" alt="LDRQ 封面：宇树 R1 真机与 Isaac Lab 仿真"></a>
  <br><em>点击观看 3 分钟介绍视频（<a href="docs/media/ldrq_pitch.mp4">docs/media/ldrq_pitch.mp4</a>）：训练、部署、sim2real 测量，以及真机上的英文指令</em>
</p>

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

## 版本

| 版本 | 位置 | 改了什么 | 状态 |
|---|---|---|---|
| v1 | release [v1.1.0](https://github.com/kedianwen/LDRQ/releases/tag/v1.1.0) | W04 行走策略、真机运行时、英文指令层、一条命令换策略 | 已发布 |
| v2 | release [v2.0.0](https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0)，`master` 分支 | 重训的行走策略：零指令不再往前蹭，能后退、侧行 | 已发布；2026-10-20 起部署在真机上 |
| V3 | `v3` 分支，[v3/](v3/README.md) 目录 | 第二个策略负责跳舞；有人靠近时交给 v2 后退 | 开发中；2026-10-24 已在真机吊装状态下运行 |

**v2：站得住、能后退**（[docs/v2_policy.md](docs/v2_policy.md)）。v1 在零指令下每 10 秒往前蹭约 0.45 m：它的训练指令范围只有前进，零正好在边界上，而且只有 2% 的训练环境给过零指令。v2 在同样的观测和动作接口上分两阶段重训：

1. 指令范围扩到包含零（后退到 −0.6 m/s、横移 ±0.5 m/s、转向 ±1.57 rad/s），15% 的环境给零指令；
2. 步态成形之后，再加上对漂移（约 1 秒平均速度）的惩罚。一开始就加的话，策略学会的是干脆站着不动、忽略小指令。

仿真结果：往前蹭降到每 10 秒 0.14 m；能后退、侧行，转向比 v1 快三倍，零摔倒。v2 走同一条一键安装流程（在 Orin 上建引擎、parity 通过才算装好）；真机上能后退、侧行，零指令时站定原地踏步。策略以上的部分不变，只是指令层读取 v2 更宽的指令范围。

**V3：跳舞，以及见人后退**（[v3/README.md](v3/README.md)）。Orin 上同一个进程里并排运行两个策略，每个控制周期都推理：

- **v2** 负责行走；
- **舞蹈策略**（[v3/models/dance_wave](v3/models/dance_wave/README.md)），用 BeyondMimic 的动作跟踪方法训练，原地跟跳一段 10 秒的舞。

硬件桥里的状态机（指挥器）决定哪个策略的输出送到电机：

- `ask "dance for 20 seconds"`：只有 1 m 内没人、机器人已站稳 1 秒才开始跳；
- 正常结束要等到小节结尾；
- 跳舞时有人走到 0.9 m 内，**同一个控制周期**切回 v2，后退 1.5 秒（约 0.33 m）后站住，直到操作者确认新的计划；
- 行走和站立时也有同样的反射，触发距离 0.6 m。

人的距离由检测框的大小估计：头部相机自带的深度在近处会被填补，读出并不存在的障碍。用左右两目自己算深度（用于障碍物）正在做。V3 的代码全部在 `v3/` 下，v1、v2 保持发布时的样子，可以原样复现。

## 系统架构

![系统架构（含 V3）](docs/system_architecture_v3.svg)

上图是 V3 的系统。v2 的系统就是去掉舞蹈策略和指挥器之后的样子（[v2 时期的架构图](docs/system_architecture.svg)）。

一句英文指令只在机器人动之前被转换成计划，一共三步：

1. 模型只负责把句子转写成结构化步骤；
2. 确定性代码对照策略的训练范围检查计划；
3. 操作者确认后机器人才执行。

计划下方还有三层，按各自的频率运行，各带安全检查：指令层 10 Hz、策略 50 Hz、电机回路 500 Hz。详细的数据流和逻辑图见 [docs/system_architecture.md](docs/system_architecture.md)。

**视觉模块**（[perception/](perception/README.md)，[docs/stageE_perception.md](docs/stageE_perception.md)；V3 的分叉 [v3/perception/](v3/perception/README.md)）。R1 EDU 的 Orin 上没有直接接相机，头部的双目模组通过机器人内部网络把画面推给 Orin。`r1_perception` 节点分四步工作：

1. 接收左目的 H264 视频流（UDP RTP，端口 5002，544×448，约 10 Hz），用 Jetson 的硬件解码器解码；
2. 每一帧用 YOLOX-Nano 做目标检测。引擎是在 Orin 上建的 TensorRT FP16：每帧 3.7 ms，检测结果和 FP32 一致。检测器和行走策略共用 GPU，两者同时运行时策略推理的 p99 仍在 0.6 ms 以内；
3. 同一目标要连续几帧都检测到才算确认；
4. 发布两个话题：
   - `/r1_perception/detections`（JSON）：每个物体的方位和距离，供 `turn_to`（"转向椅子"）和 `walk_to`（"走到椅子前 1.5 m 停下"）使用；另外，有人靠近或相机没有画面时，会让正在走的步骤停下；
   - `/r1_perception/proximity`：V3 的近距后退反射看的就是它。

人的距离按检测框的高度估计，这个比例不用卷尺标定：帮手先站在一臂远处，再往后退一步，各拍一次。模组自带的深度在近处不能用，它会把匹配不上的像素填成近距离。用左右两目自己计算深度的工作还在进行中。

现状：
- 近距后退反射已在真机上运行（V3，2026-10-24）；
- 检测引擎、parity、与策略共用 GPU 的测试都已在真机通过；
- `turn_to` / `walk_to` 已在开发机上测试，真机测试还没做。

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
  - 端侧 [release](https://github.com/kedianwen/LDRQ/releases) 附中文安装说明。

## 快速上手

```bash
git clone https://github.com/kedianwen/LDRQ.git && cd LDRQ
./run_tests.sh                                    # 不需要 GPU、ROS 或机器人，约 3 秒
python3 mission_ctl/r1_mission_cli.py capability  # 机器人自述能做什么
python3 mission_ctl/r1_mission_cli.py run "walk 3s@0.3; turn left 90" --dry-run   # 编译、检查并仿真一个计划
```

- 想在真机上运行：下载 [Releases](https://github.com/kedianwen/LDRQ/releases) 里的 `release_v2.0.0.tar.gz` 和模型包，按其中的 [中文说明](release/README_zh.md) 操作。
- 想做训练或仿真：见英文 README 的 [*Reproducing*](README.md#reproducing) 一节。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/technical_report.md](docs/technical_report.md) | 技术报告：整个项目的完整叙述 |
| [docs/v2_policy.md](docs/v2_policy.md) | v2 策略：v1 零指令往前蹭的原因、两阶段重训、后退与横移、真机部署 |
| [v3/README.md](v3/README.md) | V3：舞蹈策略、两个策略的交接、近距后退反射、真机结果（英文） |
| [docs/system_architecture.md](docs/system_architecture.md) | LLM 与运控的完整数据流、架构图和逻辑图 |
| [docs/stageA_kp_sweep.md](docs/stageA_kp_sweep.md) | 核心实验：`kp_scale` 稳定域，仿真与真机画在同一张图上 |
| [docs/project_history.md](docs/project_history.md) | 项目过程：周计划、验收门槛（PG/FR 编号的含义），以及每周的发现 |

许可证：Apache-2.0。R1 的 URDF 和网格文件来自宇树，采用 BSD-3-Clause，详见 [NOTICE](NOTICE)。
