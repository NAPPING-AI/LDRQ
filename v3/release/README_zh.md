# R1 机器人软件 v3.0.0（V3）：跳舞，以及有人靠近时后退

这个压缩包包含 V3 在**机器人上**（宇树 R1 EDU，机载 Jetson Orin NX）运行所需的全部内容。English: [README.md](README.md)。

V3 在 v2 行走系统的基础上加了两样东西：
- **跳舞。** `ask "dance for 20 seconds"` 会启动第二个策略，它被训练成原地跟随一段 10 秒的舞蹈：先右臂挥手，再左臂，然后双臂上举、屈膝两次，最后站定。双脚始终着地。
- **近距反射。** 有人进入机器人 **0.6 米**内（跳舞时 **0.9 米**），v2 行走策略在同一个控制步接管，后退 1.5 秒（约 0.33 米），然后站住，直到操作员确认新计划。

v2 原有的功能照常可用：
- 向前、后退、横移和原地转向；
- 用英文下指令：由机器人 CPU 上的小语言模型转写，代码检查，操作员确认后机器人才动。

源码、设计说明和结果：<https://github.com/NAPPING-AI/LDRQ> 的 `v3` 分支（`v3/README.md` 是详细说明）。

> **状态：预发布。**
> - V3 **只在吊装下**跑过（2026-10-24），输出关和电机输出开都跑过；还没有脱吊装运行。
> - 反射**只对人起作用**，不管物体。
> - 相机只朝前看，机器人是往它看不到的地方后退的。
>
> **机器人动之前先读[安全](#安全)一节。** 这是研究代码，机器人会摔倒，也可能伤人。

## 内容

| 发布文件 | 大小 | 内容 |
|---|---|---|
| `release_v3.0.0.tar.gz` | 约 0.5 GB | 本说明、`install.sh`、代码、两个行走策略包（v2；以及用于回滚的 v1）、跳舞策略、YOLOX-Nano、JetPack 5 版 Ollama 模型服务 |
| `qwen3-1.7b_r1-robot-v3.0.0.tar.gz` | 约 1.3 GB | 语言模型 qwen3:1.7b（Q4_K_M），Ollama 格式 |
| `SHA256SUMS` | — | 两个压缩包的校验和 |

两个压缩包解压到同一个目录 `r1-robot-v3.0.0/`：

```
r1-robot-v3.0.0/
  README.md, README_zh.md       本说明（英文 / 中文）
  install.sh                    把 V3 装到下面的 v3/，分 5 步，每步都有检查
  selftest.sh                   不会让机器人动的自检
  base/                         V3 从它安装的 v2 目录树，也是回滚用的：
    deploy/                       C++/ROS 2 硬件桥和 TensorRT 策略节点（在机器人上编译）
    bundles/                      v2（2026-10-16_17-57-14_v2_drift_ft）和 v1（回滚用）行走策略
    policy_pack/                  安装策略包：生成配置、编译、构建引擎、parity 检查
    mission_ctl/                  v2 的指令层
    perception/                   视觉节点；models/ 里有 YOLOX-Nano 和它的 parity 夹具
    llm/                          ollama_ctl.sh、Ollama 运行时、模型（来自第二个压缩包）
  v3_package/                   V3：install_v3.sh、V3 代码（硬件桥、策略节点、mission_ctl、
                                感知、反射、舞蹈）、跳舞策略、哈希清单
  v3/                           由 install.sh 生成：实际运行的 V3 目录树
  LICENSE, NOTICE, THIRD_PARTY_NOTICES.md
```

| 策略 | 做什么 | 接口 |
|---|---|---|
| `2026-10-16_17-57-14_v2_drift_ft`（v2，行走） | 向前最快 1.0 m/s，后退最快 0.6 m/s，横移 ±0.5 m/s，转向最快 1.57 rad/s；零指令时原地踏步 | 425 → 24 个关节目标，50 Hz |
| `2026-10-23_16-51-44_wave`（跳舞） | 原地循环跳那段 10 秒的舞蹈；只在小节结尾交回 v2 | 137 → 24 个关节目标，50 Hz，关节和增益与 v2 相同 |

## 系统要求

| | |
|---|---|
| 机器人 | **宇树 R1 EDU**，机载 Jetson Orin NX 16 GB，带头部双目相机模组 |
| 系统 | **JetPack 5.1.1**（L4T R35.3.1）：TensorRT 8.5.2、CUDA 11.4、Ubuntu 20.04、带 numpy 的 Python 3.8；g++ |
| ROS | **ROS 2 foxy**，装在 `/opt/ros/foxy` |
| 宇树 SDK | **unitree_sdk2** 编译安装在 `/usr/local`（`/usr/local/include/unitree/`、`/usr/local/lib/libunitree_sdk2.a`）。硬件桥要和它一起编译，本 release 不包含它 |
| 宇树 APP | 装在手机上：每次开机后都要用它打开头部相机的视频流 |
| 磁盘 / 内存 | 约 6 GB 空闲磁盘；加载模型时约 2.5 GB 空闲内存 |
| 网络 | 机器人不需要联网；需要一台电脑下载 release，再通过 SSH 拷过去 |
| 人和设备 | 一名手拿急停的操作员；一名负责走近机器人的帮手；吊架；机器人身后清空 1 米，前方清空 2.5 米 |

## 安装

**与其他目录并存。** V3 装在自己的目录 `~/r1-robot-v3.0.0` 里。以前的 release 或开发安装留下的目录（`~/r1-robot-v1.1.0`、`~/r1-robot-v2.0.0`、`~/r1-robot-v3`）保持原样。安装前先停掉所有正在运行的栈和模型服务，例如 `bash ~/r1-robot-v2.0.0/llm/ollama_ctl.sh stop`，因为它们用同一个端口。

**1. 电脑上：下载并校验。**
```bash
sha256sum -c SHA256SUMS          # 两个压缩包都应显示 OK
scp release_v3.0.0.tar.gz qwen3-1.7b_r1-robot-v3.0.0.tar.gz unitree@<机器人IP>:~/
```
要用 `.tar.gz` 拷贝，不要用 zip：zip 会丢掉文件的执行权限。

**2. 机器人上：把两个压缩包解压到同一个位置，然后自检。**
```bash
cd ~
tar xzf release_v3.0.0.tar.gz
tar xzf qwen3-1.7b_r1-robot-v3.0.0.tar.gz      # 生成 r1-robot-v3.0.0/base/llm/models/
bash ~/r1-robot-v3.0.0/selftest.sh             # 每一行都应是 "ok"；机器人不会动
```
`selftest.sh` 检查以下内容：
- 每个文件与哈希清单是否一致；
- v2 目录树的测试；
- 模型和它的运行时；
- V3 自己的测试：反射和 conductor 与金标逐拍对比（含 C++）、跳舞观测、视觉代码、指令层，以及三个 dry run。

标为 `warn` 的行是关于机器人系统的（JetPack、ROS、unitree_sdk2、TensorRT）。它们不算自检失败，但安装时这四样都必须有。

`install.sh` 把 v2 策略装进 `base/` 之后，`selftest.sh` 对 `base/` 只检查文件、模型和运行时。原因是 v2 自带的英文测试用的是按 v1 速度范围冻结的评测集，会把 v2 的后退和横移报成错误。V3 自己的测试照常运行。

**3. 安装（约 15 分钟，大部分时间在构建引擎）。** 开一个**没有 source 过任何 `env.sh` 的新终端**：
```bash
bash ~/r1-robot-v3.0.0/install.sh
```
每一步打印 `[n/5]`，任何一步失败就停下。日志在 `~/orin_commissioning/v3.0.0_install_<时间>.log`。

| 步骤 | 做什么 | 应该看到 |
|---|---|---|
| 1 检查 | 终端干净、没有栈在运行、`selftest.sh` 通过 | `selftest: OK` |
| 2 v2 策略 | 用 v2 策略包运行 `base/policy_pack/install_bundle.sh`：写入 v2 的速度范围，在 `base/` 里编译硬件桥和策略节点，**在这台机器人上**构建引擎，再和策略包里的参考值对比 | `PARITY PASS`，`max_abs` 约 1e-05（门槛 1e-3） |
| 3 视觉 | 构建 YOLOX-Nano 引擎（FP16，约 7 分钟），再和它的夹具对比 | `PARITY PASS (detections gate ...)` |
| 4 V3 | `v3_package/install_v3.sh --src base --dst v3`，分 8 个小步：复制 `base/` → 把 V3 叠加上去 → 显示配置差异 → 编译 → 检查 CycloneDDS、跑 V3 的测试 → 在这里构建跳舞引擎，两个引擎都做 parity | `joint_map.hpp identical`；配置差异只有预期的那些；`CycloneDDS from /usr/local/lib: OK`；两行 `PARITY PASS`；`source tree unchanged: OK` |
| 5 总结 | V3 装在哪里，跳舞引擎的指纹 | **记下 `fnv1a=0x...`**：每次起栈都会打印它 |

- 第 2、3 步已经做过的话会自动跳过，所以第 4 步失败后重新运行 `install.sh` 没有问题。
- 第 4 步遇到已存在的 `v3/` 会拒绝执行，先把它移开：`mv ~/r1-robot-v3.0.0/v3 ~/r1-robot-v3.0.0/v3.old`。
- 第 4 步（`[5/8]`）**预期的配置差异**：
  - `mission.yaml` 新增 `reflex: on`、`reflex_obstacles: off`、`dance*` 几行，`obj_stop_min_m` 改为 0.9；
  - `perception.yaml` 新增 `proximity_*` 和 `person_box_*`；
  - `bridge.yaml` 新增 `reflex_*` 一整段。

  出现其他以 `<` 开头的行，说明 `base/` 里有本地改动，要手动抄进 `v3/` 里对应的文件。
- 引擎指纹每次构建都不一样（TensorRT 靠计时挑选 kernel），这是正常的；数值对不对以 parity 为准。

## 每次上机

### 每次开机后、机器人动之前

1. **用手持遥控切到开发者模式。** 不切的话，宇树自己的运控会和本系统同时给电机发指令。现象像是策略不好：晃动、抖动、关节异响。
2. **锁频**（需要 sudo，重启前一直有效）：`bash ~/r1-robot-v3.0.0/v3/deploy/tools/w08_preflight.sh --lock-clocks`，必须报告 0 FAIL。
3. **打开头部相机的视频流**：在宇树 APP 里**关 video_hub**、**开 Stereo patch PC1**。APP 每次开机都会忘掉这个设置。
4. 机器人吊上吊架，脚着地，绳子留一点松弛。

### 每个终端都先执行

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # 必须在 source env.sh 之前，每个终端都要
source ~/r1-robot-v3.0.0/v3/deploy/env.sh
R=~/r1-robot-v3.0.0/v3; M=$R/mission_ctl/r1_mission_cli.py; PC=$R/perception/r1_perception_cli.py
PM=$R/perception/models; D=$R/v3_models/dance_wave; O=~/orin_commissioning; mkdir -p $O
```
`ROS_DOMAIN_ID` 设置不同的终端之间互相看不到。现象是栈好像对所有指令都没反应。

### 检查相机画面

```bash
pgrep -af stereo_capture                                  # 应该什么都不打印；有的话 kill <PID>
python3 $R/perception/tools/stereo_capture.py --probe     # 3 秒：5002、5003 两个端口都应约 10 fps
```
如果显示 0 个包：在 APP 里重新设置（关 video_hub、开 Stereo patch PC1），再 probe 一次。没有画面就不要往下做：
- 反射打开时，没有画面就不能向前走，也不能跳舞；
- 2026-10-24 有一次上机就是因为没有画面，最后进了 `DEGRADED`。

### 起服务

| 终端 | 做什么 | 命令 |
|---|---|---|
| 1 | 语言模型 | `bash $R/llm/ollama_ctl.sh start && bash $R/llm/ollama_ctl.sh health` |
| 2 | 视觉 | `cd $R && python3 $PC run --engine $PM/yolox_fp16.plan 2>&1 \| tee $O/v3_perception.log` |
| 3 | 栈（见下） | |
| 4 | 只看关键行 | `tail -f $O/v3_stack.log \| grep --line-buffered -E "DANCE\|REFLEX:\|DEGRADED"` |
| 5 | 下指令 | `python3 $M ...` |

- **终端 1**：`health` 必须显示 `OK`。模型跑在 4 个 CPU 线程上，GPU 留给两个策略和检测器。
- **终端 2**：视觉节点的状态行应显示约 12.5 Hz。

**终端 3，栈。每次都先起输出关的栈。** 电机收不到任何指令，两个策略、conductor 和反射照常运行：
```bash
PYTHONUNBUFFERED=1 ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 \
    reflex_enable:=true dance_enable:=true \
    dance_engine:=$D/policy_fp32.plan dance_reference:=$D/dance_ref.bin \
    reflex_log:=$O/v3_reflex_log.csv 2>&1 | tee $O/v3_stack.log
```
往下做之前先确认三件事：
- 状态行是 `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`，并且没有 `DEGRADED`；
- `dance engine ... fnv1a=0x...` 那一行与安装时记下的指纹一致；
- `cmd_vel envelope (as trained): vx [-0.60, 1.00]  vy [-0.50, 0.50]  wz [-1.57, 1.57]` 显示的是 v2 的速度范围。

然后按 Ctrl-C，把同一条命令改成 `enable_output:=true` 再起一次（日志改为 `$O/v3m_stack.log`，终端 4 也要跟着改名）。先看机器人用 v2 站稳 10 秒，再下任何指令。

启动参数说明：
- **数值参数要带小数点**（`1.3`、`1.0`）。写成整数（如 `kp_scale:=1`）硬件桥启动时就会退出。
- `iface` 是机器人 `192.168.123.x` 网段的网卡名，实验室那台是 `eth10`，用 `ip -br addr` 确认。
- `kp_scale` 是两个策略共用的关节刚度倍数。v1 在真机上 1.10 到 1.50 之间走得稳；V3 的每一次运行都用 **1.3**。
- `reflex_enable` 和 `dance_enable` 默认都是 `false`。不加这两个参数，栈就是普通的 v2：没有反射，不能跳舞，`ask "dance"` 会被拒绝。`dance_enable` 必须和 `reflex_enable` 一起开。

如果状态行是 `DEGRADED`：
1. 先保存原因：`grep "DEGRADED:" $O/v3_stack.log`；
2. **输出关**时：按 Ctrl-C 重新起栈；不要用 `ros2` 命令行发 resume；
3. 重启后还是 DEGRADED：停下，把日志带回；
4. **电机输出开**时：立即急停，见[安全](#安全)一节。

## 使用

### 指令

脚本形式（`run`）不需要模型，`ask` 用英文。两者都会先打印计划，等你按 `y`。加 `--log $O/ask_log.jsonl` 会记录每条请求和对应的计划。

| 做什么 | `run` 脚本 | `ask` 示例 |
|---|---|---|
| 按时间或距离向前、后退、横移 | `walk 3s@0.2`、`walk backward 1m@0.2`、`walk left 3s@0.15` | `"Walk backward 1 meter at 0.2 meters per second."` |
| 原地转向，用 IMU 航向闭环 | `turn left 90`、`turn right 180@1.0` | `"Turn left 90 degrees."` |
| 站立（原地踏步） | `stand 5s` | `"Stand for 5 seconds."` |
| **跳舞**，原地（5–60 秒；没说时长就是 20 秒） | `dance 30s` | `"Dance for 30 seconds."` |
| 一个计划里多步 | `stand 3s; walk 1m@0.3; turn right 90` | `"Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees."` |
| 机器人能做什么（按已安装的配置生成） | `python3 $M capability` | |

指令层里也有 `turn_to` 和 `walk_to`（"转向椅子""走到椅子前 1.5 米停下"），但它们**还没在真机上跑过**。留到以后输出关的上机再试。

`ask` 按以下顺序执行：
1. 模型在 CPU 上把句子转写成步骤（约 1–3 秒）；
2. 代码对照策略的训练范围检查每一步；
3. 打印计划，包括速度、时间和角度；
4. **你按 `y` 或 `N`**，按 `y` 之前先看清计划；
5. 机器人执行；
6. 打印简短的报告，说明做了什么（如 `1. Danced on the spot for 30.4 s.`），或者为什么停下（如 `Interrupted: stepped back because of a person 0.82 m away -- standing still until a new instruction`）。

| 能做 | 会拒绝，并说明原因 |
|---|---|
| 向前最快 1.0 m/s，后退最快 0.6 m/s，横移最快 0.5 m/s（默认分别为 0.4、0.3、0.2） | 超过这些速度；一步里斜着走 |
| 原地转到指定角度，最快 1.57 rad/s | 边走边转 |
| 1 米内没人时跳舞 5 到 60 秒 | 跳舞的同时做别的（"dance while walking forward"）；反射没开时跳舞 |
| 站着不动 | 单步少于 2 秒；一个计划超过 20 步、120 秒或 30 米 |

- **每条 `ask` 都是独立的**，机器人不记得上一句。"再来一次""回去""反方向"都会被拒；每次都要说清方向和带单位的数字。
- 加 `--dry-run` 只打印计划的指令轨迹，机器人不动。

### 跳舞和反射的规则

**开始跳舞**需要同时满足：
- 反射处于 ARMED 状态；
- 最新的一帧画面里 1.0 米内没人；
- 机器人已经用 v2 站稳 1 秒。

出现以下任何一种情况，**跳舞会在下一个小节边界结束**（一个小节 2–3 秒）：
- 时间到；
- 操作员按 Ctrl-C；
- `ask` 进程停止发心跳（0.5 秒）；
- 已经跳了 63 秒；
- 相机画面过期。

**反射**每个控制步（50 Hz）运行一次：

| 状态 | 含义 |
|---|---|
| ARMED | 正常状态。指令层发的速度会送到策略。但在两种情况下禁止向前走：进入停止距离内（离人至少 1.2 米，速度越快越远）；相机画面超过 0.3 秒没更新。后退、横移和转向不受限制 |
| RETREAT | 连续 2 帧有人在 0.6 米内（跳舞时 0.9 米）：由 v2 以 (−0.3, 0, 0) m/s 后退 1.5 秒。正在跳舞时，v2 在同一个控制步接管 |
| STAND | 后退之后：v2 原地踏步。要等到你确认新计划，**并且**人离开到触发距离之外，才会解除。人一直站在近处，机器人就一直站着。10 秒内最多后退 2 次，之后锁定为站立，直到下一个新计划 |

语言模型和指令层都不在这个回路里。由硬件桥自己判断，正在执行的计划会结束，并报告原因。

## 首次运行：验收检查

第一次运行、以及任何改动之后，都在吊装下按顺序做下面的检查：**A 部分输出关，B 部分电机输出开；A 部分通过了才做 B 部分。** 每一项都看终端 4 和指令的回复。

### A 部分：输出关（约 40 分钟）

| # | 做什么 | 通过 |
|---|---|---|
| A1 | 前方 2 米内没人：`python3 $M ask "dance for 10 seconds"`，按 `y` | 终端 4 依次出现 `walk -> dance, DANCING`、`ENDING`、`dance -> walk, OFF (danced; back to standing)`；回复是 `1. Danced on the spot for 1x.x s.` 和 `Done.` |
| A2 | 做 3 次，每次间隔 15 秒：`ask "dance for 30 seconds"`。帮手一开始站在正前方 2.5 米，跳起来以后**从正前方**慢慢走到一臂远 | 同一时刻出现 `DANCE: policy dance -> walk ... stepped back` 和 `REFLEX: stepping back: person too close -- person 0.xx m`；回复是 `Interrupted: stepped back because of a person ...` |
| A3 | 帮手站在一臂远不动：`ask "dance"` | 被拒：`1. Did not dance. Interrupted: the bridge did not dance: ...too close` |
| A4 | `ask "dance for 30 seconds"`，跳起来几秒后按 Ctrl-C | 出现 `ENDING`，3 秒内出现 `dance -> walk`；回复里有 `Interrupted: operator interrupt (Ctrl-C)` |
| A5 | `run "walk 3s@0.2"`、`run "walk backward 2s@0.2"`、`run "turn left 90"` | 每条都是 `Done.` |

A2 每次都要记下 `REFLEX` 那一行里的距离。**通过条件：**
- A1、A3、A4、A5 都正确；
- A2 三次都是同一时刻交接，回复正确；
- 全程没有 Traceback。

### B 部分：吊装、电机输出开（约 60 分钟）

吊着时脚上受力不足，跳舞看起来会和落地时不同。这一部分**只看安全**：不摔，交接时腿不乱蹬，绳子不会突然吃力。

| # | 做什么 | 通过 |
|---|---|---|
| B1 | `ask "dance for 30 seconds"`，做 3 次，每次之间站 15 秒以上 | 每次 30–33 秒后回到站立；没有脚滑、腿软，绳子没有突然吃力 |
| B2 | 跳舞中途用假相机给一个"人在 0.5 米"，做 10 次（方法见下） | 每次都在那个时刻交给 v2，后退约 1.5 秒后站住；0 次异常 |
| B3 | 帮手从正前方走近，做 5 次，方法同 A2 | 同 B2；回复里有 `stepped back because of a person` |
| B4 | 跳舞时按 Ctrl-C，做 2 次 | 出现 `ENDING`，3 秒内在小节结尾回到站立 |

**B2 不需要真人。**
1. 先在终端 2 按 Ctrl-C 停掉感知：两个程序同时发近距话题会互相干扰。
2. 每一次，先在一个终端起假相机：
   ```bash
   python3 $R/v3_tools/dance_devbox_camera.py --seconds 45 --timeline "0:none,12:0.5,14:none"
   ```
   **5 秒内**在另一个终端发指令（`run` 不经过模型，`--yes` 不用确认）：
   ```bash
   python3 $M run "dance 30s" --yes
   ```
3. 到第 12 秒，假相机打印 `person 0.50 m`。这时终端 4 应同时出现交接和 `REFLEX` 两行，机器人后退。
4. 等这 45 秒结束，再做下一次。

10 次里人出现的时间分别用第 12、15、18、21、24、13、16、19、22、25 秒，让它落在舞蹈的不同位置。

假相机停了以后硬件桥收不到画面，会保持站立，这是正常的。10 次做完后，重新起感知。

## 人距标定

人的距离按 `person_box_k / (框高 / 画面高)` 估计。

`perception/config/perception.yaml` 里的 `person_box_k: 0.465` 是在实验室那台机器人吊装时标定的（相机离地约 1.1 米）。相机位置不同时要重新标定。

标定不需要卷尺：
1. 帮手面对机器人，伸直手臂，指尖刚好碰到机器人胸口：约 0.65 米；
2. 再正常后退一步：约 1.3 米。

在栈和感知都运行的状态下执行：
```bash
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 0.65 --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 1.3  --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --from-csv $O/v3_cal.csv --write
```
- `--write` 会把旧文件保留为 `perception.yaml.before_cal`；标定后要重启感知。
- `python3 $R/perception/tools/prox_watch.py` 会显示几秒内反射看到的内容。

## 安全

- **电机输出开着时，急停始终拿在操作员手上。**
- **只在吊装下运行。** V3 还没脱吊装跑过。在脱吊装的检查项写好并通过之前，不要把它从吊架上放下来。
- **机器人是往它看不到的地方后退的。** 身后保持 1 米空旷。
- **只对人起作用。** 反射看不到物体（相机的深度在近处不可靠），障碍物要靠操作员看着。
- 出现以下任何情况，**立即停止**：异响、抖动、腿乱蹬、绳子突然吃力、电机输出开时出现 `DEGRADED`。急停，记下时间和现象，不要重试，把日志带回。
- 录像如果要公开，不能有人脸，并去掉音频。

## 停止

| 操作 / 情况 | 结果 |
|---|---|
| 在 `ask` / `run` 的终端按 **Ctrl-C** | 走路时：立即发零速度，报告这一步执行到哪。跳舞时：`ENDING`，在小节结尾交回 v2（3 秒内） |
| `ask` 进程意外退出 | 0.5 秒没有心跳：跳舞在小节结尾结束；走路时速度指令在 500 ms 内降到零 |
| 有人靠得太近 | RETREAT，然后 STAND（见上） |
| 硬件桥检测到故障（传感器数据过期、某个策略不响应、目标值非有限数、控制频率连续 3 秒低于 45 Hz） | **DEGRADED**：电机转为阻尼 |
| 任何意外情况 | **急停** |

结束一次上机：
1. 确认机器人吊着、静止；
2. 依次按 Ctrl-C 停栈（终端 3）、停感知（终端 2）；
3. 执行 `bash $R/llm/ollama_ctl.sh stop`。

## 故障排查

| 现象 | 原因和处理 |
|---|---|
| `install.sh` 第 1 步报 "this terminal has sourced a tree's env.sh" | 开一个新终端，运行 `install.sh` 前什么都不要 source |
| 第 2 步报 `unitree/...: No such file` | `/usr/local` 下没装 unitree_sdk2 |
| 第 2 或第 4 步 `PARITY FAILED` | 不要运行这个引擎。检查系统是否为 JetPack 5.1.1 / TensorRT 8.5.2，并把日志带回 |
| 第 4 步报 `joint_map.hpp ... differs` | `base/` 被手动改过，或装了别的策略包：在 `base/` 里重新装 v2 策略包（第 2 步） |
| 第 4 步停在 CycloneDDS 检查 | 硬件桥链接到了错误的 DDS 库：unitree_sdk2 的 CycloneDDS 必须在 `/usr/local/lib` |
| 感知 0 帧，随后硬件桥 `DEGRADED` | 检查 APP 设置（每次开机都会恢复默认），或者 `stereo_capture.py` 还占着 5002/5003 |
| 向前走一开始就停，或者根本不走 | 反射没收到新画面：启动感知（终端 2）；或者起栈时不加 `reflex_enable` / `dance_enable`，按普通 v2 方式行走 |
| `ask "dance"` 被拒："no status from the bridge on /r1_hw_bridge/reflex" | 起栈时没加 `reflex_enable:=true` |
| 一直不开始跳舞 | 相机视野里 1 米内有人或东西，或者机器人没站稳满 1 秒 |
| 刚起栈就 `DEGRADED: joint_target stale` | 输出关时出现过一次，重启后消失。重启栈；如果再出现，把日志带回 |
| 硬件桥启动就退出：`InvalidParameterTypeException` | 某个数值启动参数写成了整数，改成 `1.3`、`1.0` |
| 指令没有效果，或 `ask` 一直等硬件桥 | 各终端的 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 不一致，或没有 source `env.sh` |
| `ask` 拒绝："another publisher on ~/cmd_vel" | 还有别的 `ask` / `run` 在运行；同一时间只能有一个进程控制机器人 |
| 模型服务起不来：端口被占 | 别的目录的服务还在运行，用那个目录的 `ollama_ctl.sh stop` 停掉 |
| `ollama_ctl.sh health` 报模型没回答 | `bash $R/llm/ollama_ctl.sh restart`（运行时偶尔启动后不响应，重启即可） |
| 晃动、抖动、关节异响 | 没切开发者模式，宇树的运控也在发电机指令 |
| `ros2 topic list` 崩溃（`bad_alloc`） | 这台机器人上 ROS foxy 与宇树 SDK 的 DDS 库冲突，是已知问题。本系统除了 `ros2 launch` 不需要 `ros2` 命令行 |

## 每次上机后带回什么

```bash
cd ~ && tar czf v3_logs_$(date +%m%d).tar.gz orin_commissioning/v3*
```
压缩包里包括：
- 安装日志；
- 栈的日志；
- 反射日志：每个控制步一行，记录状态、原因、指令和距离；
- 感知日志。

另附每项检查的记录：结果、触发距离、有什么异常、大概在什么时间。

## 回滚、卸载

- **回到 v2**：每个终端改为 `source ~/r1-robot-v3.0.0/base/deploy/env.sh`，起栈时不加 `reflex_*` / `dance_*` 参数。`base/` 是一个装好的 v2 目录树：`install.sh` 只在第 2、3 步改动它（安装 v2、构建视觉引擎），第 4 步还会检查它没有被改动。
- **回到 v1**：在 `base/` 里（source 它的 `env.sh` 之后）执行
  `bash ~/r1-robot-v3.0.0/base/policy_pack/install_bundle.sh ~/r1-robot-v3.0.0/base/bundles/2026-08-19_11-03-32_week04_nohead`。重新安装 V3 之前，要先把 v2 策略包装回去。
- **卸载**：先 `bash ~/r1-robot-v3.0.0/v3/llm/ollama_ctl.sh stop`，再 `rm -rf -- ~/r1-robot-v3.0.0`。除了 `~/orin_commissioning/` 里的日志，这个目录以外没有安装任何东西。

## 许可证

代码采用 Apache-2.0（`LICENSE`、`NOTICE`）。Ollama 运行时、它附带的 CUDA 库、Qwen3 模型和 YOLOX-Nano 各自沿用自己的许可证，详见 `THIRD_PARTY_NOTICES.md`。
