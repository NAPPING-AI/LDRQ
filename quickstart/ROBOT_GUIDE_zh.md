# v2 上机指南：从快速启动包到后退、横移

适用于**宇树 R1 EDU**（机载 Jetson Orin NX）的操作员。第一次做约 1 小时，需要两个人。English: [ROBOT_GUIDE.md](ROBOT_GUIDE.md)。

v2 的能力：
- 向前走，最快 1.0 m/s；
- **后退**，最快 0.6 m/s；
- **横移**，±0.5 m/s；
- 原地转向，最快 1.57 rad/s；
- 零指令时原地踏步，不往前漂。

压缩包里也带了 v1 策略，用于回滚（§12）。

> **安全第一。** 这是研究代码，机器人会摔倒，也可能伤人。
> - 电机输出开着时，急停遥控始终拿在手上。
> - 每次第一轮运行都要吊装，绳子留一点松弛。
> - **相机只朝前看，机器人身后和两侧没有任何东西在看着。** 后退或横移之前，身后和两侧至少清空 2 米。
> - 机器人动之前先读 §6。

## 需要准备

| | |
|---|---|
| 机器人 | 宇树 R1 **EDU**，机载 Orin NX 16 GB |
| 机器人系统 | JetPack 5.1.1（TensorRT 8.5.2、CUDA 11.4、Ubuntu 20.04、Python 3.8）；ROS 2 foxy 装在 `/opt/ros/foxy`；**unitree_sdk2** 装在 `/usr/local`（硬件桥编译时要用，压缩包里没有） |
| 磁盘 / 内存 | 约 5 GB 空闲磁盘；加载模型时约 2.5 GB 空闲内存 |
| 一台电脑 | 通过 SSH 往机器人拷文件；机器人本身不需要联网 |
| 人和设备 | 一名拿急停的操作员，一名帮手；吊架 |

**从 v1 升级：** v2 装在自己的目录 `~/ldrq-v2`，v1 的目录（`~/ldrq-v1` 或 `~/r1-robot-v1.1.0`）保持原样。先停掉 v1 的栈和它的模型服务（`bash <v1 目录>/llm/ollama_ctl.sh stop`），否则端口被占，新服务起不来。

## 1. 电脑上：把压缩包拷到机器人

先克隆本分支（`git clone -b v2 https://github.com/NAPPING-AI/LDRQ.git`），然后：

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v2-quickstart.tar.gz.sha256        # 应显示 OK
scp ldrq-v2-quickstart.tar.gz unitree@<机器人IP>:~/
```
要用 `.tar.gz` 拷贝，不要用 zip：zip 会丢掉文件的执行权限。

## 2. 机器人上：解压并自检

```bash
cd ~ && tar xzf ldrq-v2-quickstart.tar.gz             # 生成 ~/ldrq-v2/
bash ~/ldrq-v2/selftest.sh
```
每一行都应该是 `ok`，以下两种 `warn` 除外：
- 关于语言模型的两行 `warn`，做完第 3 步后会消失；
- 不在机器人上运行时，关于系统的几行 `warn`。

`selftest.sh` 不会让机器人动，也不需要起栈。它检查以下内容：
- 每个文件与 `FILES.sha256` 是否一致；
- 指令层和英文前端的测试；
- 评测集与随包配置是否一致；
- 两个策略包是否有效、坏包是否会被拒；
- 机器人系统：JetPack 5、ROS 2 foxy、unitree_sdk2、TensorRT 8。

## 3. 语言模型（`ask` 需要，`run` 不需要）

模型服务（JetPack 5 版 Ollama，约 0.5 GB）和模型（qwen3:1.7b，约 1.3 GB）太大，放不进 git。v1 和 v2 用的是同一套。三种方式选一种：

**A. 机器人上已经有 release 目录**（`~/r1-robot-v2.0.0` 或 `~/r1-robot-v1.1.0`）：直接链接它的运行时和模型。它们只会被读取，不会被改动。
```bash
R=~/r1-robot-v2.0.0                       # 或 ~/r1-robot-v1.1.0
ln -s $R/llm/ollama ~/ldrq-v2/llm/ollama
ln -s $R/llm/models ~/ldrq-v2/llm/models
```

**B. 没有**：
1. 在电脑上从 v2.0.0 的 release（<https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0>）下载 `release_v2.0.0.tar.gz`、`qwen3-1.7b_r1-robot-v2.0.0.tar.gz` 和 `SHA256SUMS`。
2. 用 `sha256sum -c SHA256SUMS` 校验，然后把两个压缩包拷到机器人上。
3. 在机器人上执行：
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v2.0.0.tar.gz r1-robot-v2.0.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v2.0.0.tar.gz
mv r1-robot-v2.0.0/llm/ollama r1-robot-v2.0.0/llm/models ~/ldrq-v2/llm/
```

**C. 不用模型**：跳过这一步和第 5 步。`run` 可以用脚本形式下同样的计划（如 `run "walk backward 1m@0.2; walk left 3s@0.15"`），`ask` 不能用。

做完后再跑一次 `bash ~/ldrq-v2/selftest.sh`，关于模型的两行 `warn` 应该变成 `ok`。

## 4. 编译并安装 v2 策略（约 2 分钟）

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v2/deploy/env.sh
bash ~/ldrq-v2/policy_pack/install_bundle.sh ~/ldrq-v2/bundles/2026-10-16_17-57-14_v2_drift_ft
```
`install_bundle.sh` 分 8 步执行，每步都有检查，任何一步失败就停下：
1. 校验策略包；
2. 显示将要改动的内容：对 v2 来说是 `bridge.yaml` 里三行 `cmd_vel_*_range`；
3. 暂存文件；
4. 生成关节映射和配置（关节映射与 v1 相同，因为增益相同）；
5. `colcon build` 编译硬件桥和策略节点（约 30 秒）；
6. **在这台机器人上**构建 TensorRT 引擎（约 25 秒）；
7. **把引擎的输出和策略包里的参考值对比**（parity）：`max_abs` 应在 1e-05 左右，门槛 1e-3；
8. 写入 `deploy/artifacts/installed.json`。

最后打印的指纹每次构建都不一样，这是正常的；数值对不对以 parity 为准。

## 5. 启动语言模型（选了方式 C 就跳过）

```bash
bash ~/ldrq-v2/llm/ollama_ctl.sh start
bash ~/ldrq-v2/llm/ollama_ctl.sh warm
# 应显示："health: OK -- qwen3:1.7b answered in X s, and the plan is right"
bash ~/ldrq-v2/llm/ollama_ctl.sh status     # 模型已加载，跑在 CPU 上
```
- `mission_ctl/config/mission.yaml` 把模型放在 4 个 CPU 线程上（`llm_num_gpu: 0`、`llm_num_thread: 4`），GPU 留给行走策略。
- 服务只监听 `127.0.0.1:11434`，以当前用户运行，不需要 sudo，也不用 systemd。
- 机器人运动时不要开 `ollama_ctl.sh watchdog`。

## 6. 每次开机后、机器人动之前

1. **用手持遥控把机器人切到开发者模式。** 不切的话，宇树自己的运控会和本系统同时给电机发指令。现象像是策略不好：晃动、抖动、关节异响。
2. **锁频**（需要 sudo，重启前一直有效）：`bash ~/ldrq-v2/deploy/tools/w08_preflight.sh --lock-clocks`，必须报告 0 FAIL。
3. 机器人吊上吊架，绳子留一点松弛；身后和两侧清空 2 米。

**每个**终端都先执行下面这段：
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # 必须在 source env.sh 之前
source ~/ldrq-v2/deploy/env.sh
M=~/ldrq-v2/mission_ctl/r1_mission_cli.py; O=~/orin_commissioning; mkdir -p $O
```
`ROS_DOMAIN_ID` 设置不同的终端之间互相看不到。现象是栈好像对所有指令都没反应。

## 7. 起栈（终端 1）

**先起输出关的栈。** 电机收不到任何指令，其他部分照常运行：
```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 2>&1 | tee $O/v2_stack_off.log
```
1. 等出现 `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`（约 5 秒），并且没有 `DEGRADED`。
2. 启动输出里的速度范围必须是 v2 的：
   `cmd_vel envelope (as trained): vx [-0.60, 1.00]  vy [-0.50, 0.50]  wz [-1.57, 1.57]`。
3. 按 Ctrl-C，改成 `enable_output:=true` 再起一次（日志改为 `$O/v2_stack_on.log`）。

注意：
- **数值参数要带小数点**（`1.3`、`1.0`、`55.0`）。写成整数（如 `kp_scale:=1`）硬件桥启动时就会退出。
- `iface` 是机器人 `192.168.123.x` 网段的网卡名，实验室那台是 `eth10`，用 `ip -br addr` 确认。
- `kp_scale`：v2 和 v1 用相同的增益训练，真机上也用 **1.3**。v2 自己的稳定范围还没扫过。

## 8. 下指令（终端 2）

用脚本，不需要模型：
```bash
python3 $M capability                                  # 机器人能做什么（按已安装的策略包生成）
python3 $M run "walk backward 1m@0.2; walk left 3s@0.15"
```
用英文（加 `--log` 会记录每条请求和对应的计划）：
```bash
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Walk backward 1 meter at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Sidestep to the left for 3 seconds." --log $O/ask_log.jsonl
python3 $M ask "Turn left 90 degrees." --log $O/ask_log.jsonl
```
`ask` 按以下顺序执行：
1. 模型在 CPU 上把句子转写成步骤（约 1–3 秒）；
2. 代码对照策略的训练范围检查每一步；
3. 打印计划，包括速度、时间和角度；
4. **你按 `y` 或 `N`**，按 `y` 之前先看清计划；
5. 机器人执行；
6. 打印一段简短的报告，说明做了什么。

| 能做 | 会拒绝，并说明原因 |
|---|---|
| 向前最快 1.0 m/s，后退最快 0.6 m/s，横移最快 0.5 m/s（默认分别为 0.4、0.3、0.2） | 超过这些速度；一步里斜着走 |
| 原地转到指定角度，最快 1.57 rad/s，用 IMU 航向闭环 | 边走边转 |
| 站着不动（原地踏步） | 单步少于 2 秒；一个计划超过 20 步、120 秒或 30 米 |
| 走一段大致的距离（时间 × 指令速度，没有里程计） | 测量实际走了多远 |

方向可以是 `forward`（默认）、`backward`、`left`、`right`；速度总是写正数。每条 `ask` 都是独立的，每次都要说清方向和带单位的数字。

**第一次运行的检查项**（吊装、输出开）：

| 检查 | 预期 |
|---|---|
| `run "stand 10s"` | 原地踏步，看不出往前漂 |
| `run "walk 3s@0.2"` | 向前走，停下，保持平衡 |
| `run "walk backward 2s@0.2"` | 往后走（吊着大约能移动 0.3 米） |
| `run "walk left 3s@0.15"`，再 `run "walk right 3s@0.15"` | 横移，两腿不交叉 |
| `run "turn left 90"` | 转向。吊着时绳子会扭，角度只在脱吊装时判断 |
| `run "walk 10s@0.2"` 时按 Ctrl-C | 立即发零速度；报告里写明走了多久 |
| `ask "Walk backward 1 meter at 0.2 meters per second."` | 按 `y` 之前打印的计划是后退、0.20 m/s、5 秒 |

## 9. 停止

| 操作 / 情况 | 结果 |
|---|---|
| 在 `ask` / `run` 的终端按 **Ctrl-C** | 立即发零速度，报告这一步执行到哪；机器人继续保持平衡 |
| `ask` 进程意外退出 | 500 ms 收不到指令，硬件桥自动发零速度 |
| 硬件桥检测到故障（传感器数据过期、策略不响应、目标值非有限数、控制频率连续 3 秒低于 45 Hz） | **DEGRADED**：电机转为阻尼。按 Ctrl-C 停栈，再重新起 |
| 任何意外情况 | **急停** |

收尾：确认机器人吊着、静止，在终端 1 按 Ctrl-C 停栈，然后执行 `bash ~/ldrq-v2/llm/ollama_ctl.sh stop`。

## 10. 视觉（可选；stage E，还没用真实画面跑过）

视觉节点和 `turn_to` / `walk_to` 两个步骤都在压缩包里，默认 `perception: off`。真机上已经通过的有：引擎、parity、与策略共用 GPU 的测试。`turn_to` 和 `walk_to` 还没在真机上跑过，所以在它们跑通之前，这一部分**只在输出关时做**。

1. **相机画面，每次开机后都要设置。** 在宇树 APP 里**关 video_hub**、**开 Stereo patch PC1**。如果要用深度（`walk_to` 需要）：`sudo unitree_depth_start`。
2. **探测**（不会让机器人动）：`cd ~/ldrq-v2 && python3 perception/r1_perception_cli.py probe`。
3. **构建引擎**（约 7 分钟；真机用的是 FP16）：
   ```bash
   B=$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
   PM=~/ldrq-v2/perception/models
   ( cd $PM && sha256sum -c SHA256SUMS && sha256sum -c yolox_fixture.sha256 )
   $B/r1_build_engine --onnx $PM/yolox_nano.onnx --plan $PM/yolox_fp16.plan --fp16
   python3 ~/ldrq-v2/perception/r1_perception_cli.py engine-check --engine $PM/yolox_fp16.plan --gate detections
   ```
   最后一行必须是 `PARITY PASS`。
4. **启动视觉节点**（终端 3，栈已经起来）：
   `cd ~/ldrq-v2 && python3 perception/r1_perception_cli.py run --engine $PM/yolox_fp16.plan 2>&1 | tee $O/v2_perception.log`。
   状态行应显示约 12.5 Hz。
5. **打开这两个步骤**：把 `~/ldrq-v2/mission_ctl/config/mission.yaml` 里的 `perception` 改成 `on`。然后在输出关的状态下试：
   - `python3 $M run "turn_to the chair"`；
   - 在正前方 3 米放一把椅子，`python3 $M run "walk_to the chair stop_at 1.5"`。

   测试流程和要记录的内容见仓库里的 [docs/stageE_perception.md](../docs/stageE_perception.md)。

## 11. 故障排查

| 现象 | 原因和处理 |
|---|---|
| `install_bundle.sh` 第 5 步报 `unitree/...: No such file` | `/usr/local` 下没装 unitree_sdk2 |
| 第 7 步 `PARITY FAILED` | 不要运行这个引擎。检查系统是否为 JetPack 5.1.1 / TensorRT 8.5.2 |
| 硬件桥启动就退出：`InvalidParameterTypeException` | 某个数值启动参数写成了整数，改成 `1.3`、`55.0` |
| 硬件桥打印的是 v1 的范围（`vx [0.00, 1.00]  vy [0.00, 0.00]  wz [-0.50, 0.50]`） | 这个目录里没装 v2 策略包，做第 4 步 |
| `ask` 拒绝后退或横移 | 同上：指令层读取的是已安装策略包的速度范围 |
| 指令没有效果，或 `ask` 一直等硬件桥 | 各终端的 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 不一致，或没有 source `env.sh` |
| `ask` 拒绝："another publisher on ~/cmd_vel" | 还有别的 `ask` / `run` 在运行；同一时间只能有一个进程控制机器人 |
| 模型服务起不来：端口被占 | 别的目录的服务还在运行，用那个目录的 `ollama_ctl.sh stop` 停掉 |
| `ollama_ctl.sh warm` 报模型没回答 | `bash ~/ldrq-v2/llm/ollama_ctl.sh restart`（运行时偶尔启动后不响应，重启即可） |
| 晃动、抖动、关节异响 | 没切开发者模式，宇树的运控也在发电机指令 |
| 视觉节点收不到画面（0 帧） | 重启后 APP 设置丢了（关 video_hub、开 Stereo patch PC1），或有别的程序占着 5002 端口 |
| `ros2 topic list` 崩溃（`bad_alloc`） | 这台机器人上 ROS foxy 与宇树 SDK 的 DDS 库冲突，是已知问题。本系统除了 `ros2 launch` 不需要 `ros2` 命令行 |

## 12. 回滚到 v1

```bash
bash ~/ldrq-v2/policy_pack/install_bundle.sh ~/ldrq-v2/bundles/2026-08-19_11-03-32_week04_nohead
```
硬件桥的速度范围恢复为 v1 的（只能向前走，转向最快 0.5 rad/s），指令层也会重新拒绝后退和横移。想回到 v2，再装一次 v2 策略包即可。

## 13. 带回什么

```bash
cd ~ && tar czf v2_logs_$(date +%m%d).tar.gz orin_commissioning/
```
另附每项检查的记录：下了什么指令、机器人做了什么、有什么异常、大概在什么时间。录像如果要公开，不能有人脸，并去掉音频。

## 卸载

```bash
bash ~/ldrq-v2/llm/ollama_ctl.sh stop
rm -rf -- ~/ldrq-v2
```
这个目录以外没有安装任何东西。如果选了方式 A，链接指向 release 目录，那个目录保持原样。
