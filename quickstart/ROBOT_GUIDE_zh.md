# v1 上机指南：从快速启动包到第一条英文指令

适用于**宇树 R1 EDU**（机载 Jetson Orin NX）的操作员。第一次做约 1 小时，需要两个人。English: [ROBOT_GUIDE.md](ROBOT_GUIDE.md)。

v1 能向前走（最快 1.0 m/s）和原地转向（最快 0.5 rad/s），不能后退或横移；后退和横移是 v2 的功能（`v2` 分支）。

> **安全第一。** 这是研究代码，机器人会摔倒，也可能伤人。
> - 电机输出开着时，急停遥控始终拿在手上。
> - 每次第一轮运行都要吊装，绳子留一点松弛。
> - 机器人动之前先读 §6。

## 需要准备

| | |
|---|---|
| 机器人 | 宇树 R1 **EDU**，机载 Orin NX 16 GB |
| 机器人系统 | JetPack 5.1.1（TensorRT 8.5.2、CUDA 11.4、Ubuntu 20.04、Python 3.8）；ROS 2 foxy 装在 `/opt/ros/foxy`；**unitree_sdk2** 装在 `/usr/local`（硬件桥编译时要用，压缩包里没有） |
| 磁盘 / 内存 | 约 5 GB 空闲磁盘；加载模型时约 2.5 GB 空闲内存 |
| 一台电脑 | 通过 SSH 往机器人拷文件；机器人本身不需要联网 |
| 人和设备 | 一名拿急停的操作员，一名帮手；吊架 |

## 1. 电脑上：把压缩包拷到机器人

先克隆本分支（`git clone -b main https://github.com/NAPPING-AI/LDRQ.git`），然后：

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v1-quickstart.tar.gz.sha256        # 应显示 OK
scp ldrq-v1-quickstart.tar.gz unitree@<机器人IP>:~/
```
要用 `.tar.gz` 拷贝，不要用 zip：zip 会丢掉文件的执行权限。

## 2. 机器人上：解压并自检

```bash
cd ~ && tar xzf ldrq-v1-quickstart.tar.gz             # 生成 ~/ldrq-v1/
bash ~/ldrq-v1/selftest.sh
```
每一行都应该是 `ok`，以下两种 `warn` 除外：
- 关于语言模型的两行 `warn`，做完第 3 步后会消失；
- 不在机器人上运行时，关于系统的几行 `warn`。

`selftest.sh` 不会让机器人动，也不需要起栈。它检查以下内容：
- 每个文件与 `FILES.sha256` 是否一致；
- 指令层和英文前端的测试；
- 评测集与随包配置是否一致；
- 策略包是否有效、坏包是否会被拒；
- 机器人系统：JetPack 5、ROS 2 foxy、unitree_sdk2、TensorRT 8。

## 3. 语言模型（`ask` 需要，`run` 不需要）

模型服务（JetPack 5 版 Ollama，约 0.5 GB）和模型（qwen3:1.7b，约 1.3 GB）太大，放不进 git。三种方式选一种：

**A. 机器人上已经装过 v1.1.0**（`~/r1-robot-v1.1.0`）：直接链接它的运行时和模型。它们只会被读取，不会被改动。
```bash
ln -s ~/r1-robot-v1.1.0/llm/ollama ~/ldrq-v1/llm/ollama
ln -s ~/r1-robot-v1.1.0/llm/models ~/ldrq-v1/llm/models
```

**B. 没装过**：
1. 在电脑上从 v1.1.0 的 release（<https://github.com/kedianwen/LDRQ/releases/tag/v1.1.0>）下载 `release_v1.1.0.tar.gz`、`qwen3-1.7b_r1-robot-v1.1.0.tar.gz` 和 `SHA256SUMS`。
2. 用 `sha256sum -c SHA256SUMS` 校验，然后把两个压缩包拷到机器人上。
3. 在机器人上执行：
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v1.1.0.tar.gz r1-robot-v1.1.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v1.1.0.tar.gz
mv r1-robot-v1.1.0/llm/ollama r1-robot-v1.1.0/llm/models ~/ldrq-v1/llm/
```

**C. 不用模型**：跳过这一步和第 5 步。`run` 可以用脚本形式下同样的计划（如 `run "walk 3s@0.2; turn left 90"`），`ask` 不能用。

做完后再跑一次 `bash ~/ldrq-v1/selftest.sh`，关于模型的两行 `warn` 应该变成 `ok`。

## 4. 编译并安装策略（约 2 分钟）

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v1/deploy/env.sh
bash ~/ldrq-v1/policy_pack/install_bundle.sh ~/ldrq-v1/bundles/2026-08-19_11-03-32_week04_nohead
```
`install_bundle.sh` 分 8 步执行，每步都有检查，任何一步失败就停下：
1. 校验策略包；
2. 显示将要改动的内容；
3. 暂存文件；
4. 生成关节映射和配置；
5. `colcon build` 编译硬件桥和策略节点（约 30 秒）；
6. **在这台机器人上**构建 TensorRT 引擎（约 25 秒）；
7. **把引擎的输出和策略包里的参考值对比**（parity）：`max_abs` 应在 1e-05 左右，门槛 1e-3；实验室的机器人测得 1.144e-05；
8. 写入 `deploy/artifacts/installed.json`。

最后几行会打印引擎路径和指纹。每次构建的指纹都不一样（TensorRT 靠计时挑选 kernel），这是正常的；数值对不对以 parity 为准。

## 5. 启动语言模型（选了方式 C 就跳过）

```bash
bash ~/ldrq-v1/llm/ollama_ctl.sh start
bash ~/ldrq-v1/llm/ollama_ctl.sh warm
# 应显示："health: OK -- qwen3:1.7b answered in X s, and the plan is right"
bash ~/ldrq-v1/llm/ollama_ctl.sh status     # 应显示模型 "on GPU"（约 1.5 GB）
```
- v1 的模型跑在 Orin 的 GPU 上（`mission_ctl/config/mission.yaml` 里的 `llm_num_gpu: -1`）。
- 服务只监听 `127.0.0.1:11434`，以当前用户运行，不需要 sudo，也不用 systemd。
- 如果端口已被别的服务占用（比如旧目录启动的），先停掉那个，例如 `bash ~/r1-robot-v1.1.0/llm/ollama_ctl.sh stop`。
- 机器人运动时不要开 `ollama_ctl.sh watchdog`。

## 6. 每次开机后、机器人动之前

1. **用手持遥控把机器人切到开发者模式。** 不切的话，宇树自己的运控会和本系统同时给电机发指令。现象像是策略不好：晃动、抖动、关节异响。
2. **锁频**（需要 sudo，重启前一直有效）：`bash ~/ldrq-v1/deploy/tools/w08_preflight.sh --lock-clocks`，必须报告 0 FAIL。
3. 机器人吊上吊架，绳子留一点松弛；周围 1 米内清空。

**每个**终端都先执行下面这段：
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1      # 必须在 source env.sh 之前
source ~/ldrq-v1/deploy/env.sh
M=~/ldrq-v1/mission_ctl/r1_mission_cli.py; O=~/orin_commissioning; mkdir -p $O
```
`ROS_DOMAIN_ID` 设置不同的终端之间互相看不到。现象是栈好像对所有指令都没反应。

## 7. 起栈（终端 1）

**先起输出关的栈。** 电机收不到任何指令，其他部分照常运行：
```bash
ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 2>&1 | tee $O/v1_stack_off.log
```
1. 等出现 `RUNNING | obs 50.0 Hz ... cmd 500.0 Hz`（约 5 秒），并且没有 `DEGRADED`。
2. 按 Ctrl-C，改成 `enable_output:=true` 再起一次（日志改为 `$O/v1_stack_on.log`）。
3. 机器人进入策略的站立姿态，原地踏步。

注意：
- **数值参数要带小数点**（`1.3`、`1.0`、`55.0`）。写成整数（如 `kp_scale:=1`）硬件桥启动时就会退出。
- `iface` 是机器人 `192.168.123.x` 网段的网卡名，实验室那台是 `eth10`，用 `ip -br addr` 确认。
- `kp_scale` 是关节刚度的倍数。真机在 1.10 到 1.50 之间走得稳，训练值 1.0 正好在边缘，所以用 **1.3**。

## 8. 下指令（终端 2）

用脚本，不需要模型：
```bash
python3 $M capability                         # 机器人能做什么（按已安装的策略包生成）
python3 $M run "walk 3s@0.2; turn left 90"     # 显示计划，等你按 y/N
```
用英文（加 `--log` 会记录每条请求和对应的计划）：
```bash
python3 $M ask "Walk forward for 3 seconds at 0.2 meters per second." --log $O/ask_log.jsonl
python3 $M ask "Turn left 90 degrees." --log $O/ask_log.jsonl
python3 $M ask "Stand for 3 seconds, then walk forward 1 meter at 0.3 meters per second, then turn right 90 degrees." --log $O/ask_log.jsonl
```
`ask` 按以下顺序执行：
1. 模型把句子转写成步骤（约 1–4 秒）；
2. 代码对照策略的训练范围检查每一步；
3. 打印计划，包括速度、时间和角度；
4. **你按 `y` 或 `N`**，按 `y` 之前先看清计划；
5. 机器人执行；
6. 打印一段简短的报告，说明做了什么。

| 能做 | 会拒绝，并说明原因 |
|---|---|
| 向前走，0 到 1.0 m/s（默认 0.4） | 后退、横移 |
| 原地转到指定角度，最快 0.5 rad/s，用 IMU 航向闭环 | 边走边转 |
| 站着不动 | 单步少于 2 秒；一个计划超过 20 步、120 秒或 30 米 |
| 走一段大致的距离（时间 × 指令速度，没有里程计） | 测量实际走了多远 |

- **每条 `ask` 都是独立的**，机器人不记得上一句。"再来一次""回去""反方向"都会被拒；每次都要说清方向和带单位的数字。
- 加 `--dry-run` 只仿真并打印指令轨迹，机器人不动。

**第一次运行的检查项**（吊装、输出开）：

| 检查 | 预期 |
|---|---|
| `run "stand 5s"` | 原地踏步，然后打印报告 |
| `run "walk 3s@0.2"` | 向前走，然后停下并保持平衡 |
| `run "turn left 90"` | 转向。吊着时绳子会扭，角度只在脱吊装时判断 |
| `run "walk 10s@0.2"` 时按 Ctrl-C | 立即发零速度；报告里写明走了多久 |
| `ask "Walk forward for 3 seconds at 0.2 meters per second."` | 按 `y` 之前打印的计划是 3 秒、0.20 m/s |

## 9. 停止

| 操作 / 情况 | 结果 |
|---|---|
| 在 `ask` / `run` 的终端按 **Ctrl-C** | 立即发零速度，报告这一步执行到哪；机器人继续保持平衡 |
| `ask` 进程意外退出 | 500 ms 收不到指令，硬件桥自动发零速度 |
| 硬件桥检测到故障（传感器数据过期、策略不响应、目标值非有限数、控制频率连续 3 秒低于 45 Hz） | **DEGRADED**：电机转为阻尼。按 Ctrl-C 停栈，再重新起 |
| 任何意外情况 | **急停** |

收尾：确认机器人吊着、静止，在终端 1 按 Ctrl-C 停栈，然后执行 `bash ~/ldrq-v1/llm/ollama_ctl.sh stop`。

## 10. 故障排查

| 现象 | 原因和处理 |
|---|---|
| `install_bundle.sh` 第 5 步报 `unitree/...: No such file` | `/usr/local` 下没装 unitree_sdk2 |
| 第 7 步 `PARITY FAILED` | 不要运行这个引擎。检查系统是否为 JetPack 5.1.1 / TensorRT 8.5.2 |
| 硬件桥启动就退出：`InvalidParameterTypeException` | 某个数值启动参数写成了整数，改成 `1.3`、`55.0` |
| 指令没有效果，或 `ask` 一直等硬件桥 | 各终端的 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 不一致，或没有 source `env.sh` |
| `ask` 拒绝："another publisher on ~/cmd_vel" | 还有别的 `ask` / `run` 在运行；同一时间只能有一个进程控制机器人 |
| `ollama_ctl.sh warm` 报模型没回答 | `bash ~/ldrq-v1/llm/ollama_ctl.sh restart`（运行时偶尔启动后不响应，重启即可） |
| 晃动、抖动、关节异响 | 没切开发者模式，宇树的运控也在发电机指令 |
| `ros2 topic list` 崩溃（`bad_alloc`） | 这台机器人上 ROS foxy 与宇树 SDK 的 DDS 库冲突，是已知问题。本系统除了 `ros2 launch` 不需要 `ros2` 命令行 |

## 11. 带回什么

```bash
cd ~ && tar czf v1_logs_$(date +%m%d).tar.gz orin_commissioning/
```
另附每项检查的记录：下了什么指令、机器人做了什么、有什么异常、大概在什么时间。录像如果要公开，不能有人脸，并去掉音频。

## 卸载

```bash
bash ~/ldrq-v1/llm/ollama_ctl.sh stop
rm -rf -- ~/ldrq-v1
```
这个目录以外没有安装任何东西。如果选了方式 A，链接指向 `~/r1-robot-v1.1.0`，那个目录保持原样。
