# V3 上机指南：从快速启动包到跳舞和近距后退

适用于**宇树 R1 EDU**（机载 Jetson Orin NX）的操作员。English: [ROBOT_GUIDE.md](ROBOT_GUIDE.md)。

| 部分 | 时间 |
|---|---|
| 安装 | 约 1 小时 |
| 输出关的检查 | 约 40 分钟 |
| 吊装、电机输出开的检查 | 约 60 分钟 |

需要两个人：一名拿急停的操作员，一名走近机器人的帮手。

V3 = v2 + 跳舞策略 + 近距反射：
- `ask "dance for 20 seconds"` 让机器人原地跳舞（1 米内没人时才会开始）；
- 跳舞时有人进入 **0.9 米**内（走路或站立时是 0.6 米），v2 在同一个控制步接管，后退 1.5 秒（约 0.33 米）后站住。

本指南沿用 2026-10-24 在真机上实际跑过的流程，只是从零开始安装。

> **安全第一。** 这是研究代码，机器人会摔倒，也可能伤人。
> - 电机输出开着时，急停遥控始终拿在手上。
> - 本指南的所有运行都**在吊装下**进行：V3 还没做过脱吊装测试。
> - 机器人是往**相机看不到的地方**后退的：身后清空 1 米。
> - 反射**只对人起作用**，不管物体；障碍物要靠操作员看着。
> - 出现以下任何情况，**马上急停**，记下时间和现象，不要重试，本次上机到此为止：异响、抖动、腿乱蹬、绳子突然吃力、电机输出开时出现 `DEGRADED`。

## 需要准备

| | |
|---|---|
| 机器人 | 宇树 R1 **EDU**，机载 Orin NX 16 GB，头部双目模组 |
| 机器人系统 | JetPack 5.1.1（TensorRT 8.5.2、CUDA 11.4、Ubuntu 20.04、带 numpy 的 Python 3.8）；ROS 2 foxy 装在 `/opt/ros/foxy`；**unitree_sdk2** 装在 `/usr/local`（压缩包里没有） |
| 宇树 APP | 装在手机上，每次开机后用它打开相机视频流 |
| 磁盘 / 内存 | 约 6 GB 空闲磁盘；加载模型时约 2.5 GB 空闲内存 |
| 一台电脑 | 通过 SSH 往机器人拷文件；机器人本身不需要联网 |
| 人和设备 | 拿急停的操作员、帮手；吊架；机器人身后清空 1 米，前方清空 2.5 米 |

## A 部分：安装（只做一次）

### A1. 电脑上：把压缩包拷到机器人

先克隆本分支（`git clone -b v3 https://github.com/NAPPING-AI/LDRQ.git`），然后：

```bash
cd LDRQ/quickstart
sha256sum -c ldrq-v3-quickstart.tar.gz.sha256        # 应显示 OK
scp ldrq-v3-quickstart.tar.gz unitree@<机器人IP>:~/
```
要用 `.tar.gz` 拷贝，不要用 zip：zip 会丢掉文件的执行权限。

### A2. 机器人上：解压，给底座做自检

开一个**新终端，不要 source 任何 `env.sh`**，也不要起栈：
```bash
cd ~ && tar xzf ldrq-v3-quickstart.tar.gz      # 生成 ~/ldrq-v3/{base,v3_package}
cat ~/ldrq-v3/VERSION ~/ldrq-v3/v3_package/VERSION
bash ~/ldrq-v3/base/selftest.sh
```
- `base/` 是 v2 的机器人目录树，V3 从它安装。
- `v3_package/` 是 V3 本身和它的安装程序。
- 自检的每一行都应是 `ok`。关于语言模型的两行 `warn` 例外，做完 A3 后会消失。

### A3. 语言模型（`ask` 需要，`run` 不需要）

模型服务（JetPack 5 版 Ollama）和模型（qwen3:1.7b）共约 1.8 GB，放不进 git。必须**在 A6 之前**放进 `base/`，因为 V3 的目录树会从这里链接它们。三种方式选一种：

**A. 机器人上已经有 release 目录**（`~/r1-robot-v2.0.0` 或 `~/r1-robot-v1.1.0`）：
```bash
R0=~/r1-robot-v2.0.0                      # 或 ~/r1-robot-v1.1.0
ln -s $R0/llm/ollama ~/ldrq-v3/base/llm/ollama
ln -s $R0/llm/models ~/ldrq-v3/base/llm/models
```
**B. 没有**：
1. 在电脑上从 <https://github.com/kedianwen/LDRQ/releases/tag/v2.0.0> 下载 `release_v2.0.0.tar.gz`、`qwen3-1.7b_r1-robot-v2.0.0.tar.gz` 和 `SHA256SUMS`。
2. 用 `sha256sum -c SHA256SUMS` 校验，把两个压缩包拷到机器人上。
3. 在机器人上执行：
```bash
mkdir -p ~/llm_tmp && cd ~/llm_tmp
tar xzf ~/release_v2.0.0.tar.gz r1-robot-v2.0.0/llm/ollama
tar xzf ~/qwen3-1.7b_r1-robot-v2.0.0.tar.gz
mv r1-robot-v2.0.0/llm/ollama r1-robot-v2.0.0/llm/models ~/ldrq-v3/base/llm/
```
**C. 不用模型**：`run` 可以用脚本形式下同样的计划（`run "dance 20s"`），`ask` 不能用。

所有模型服务都用 11434 端口，起之前先停掉别的服务。

### A4. 在底座里安装 v2 策略（约 2 分钟）

```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/ldrq-v3/base/deploy/env.sh
bash ~/ldrq-v3/base/policy_pack/install_bundle.sh ~/ldrq-v3/base/bundles/2026-10-16_17-57-14_v2_drift_ft
```
- 共 8 步，每步都有检查，任何一步失败就停下。
- 第 7 步是 parity：`max_abs` 应在 1e-05 左右，门槛 1e-3。
- 第 4 步把 v2 的速度范围写进底座的 `bridge.yaml`。

### A5. 在底座里构建视觉引擎（约 7 分钟）

在同一个终端里执行：
```bash
B=$R1_DEPLOY_ROOT/ros2_ws/install/r1_policy_runner/lib/r1_policy_runner
PM=~/ldrq-v3/base/perception/models
( cd $PM && sha256sum -c SHA256SUMS && sha256sum -c yolox_fixture.sha256 )
$B/r1_build_engine --onnx $PM/yolox_nano.onnx --plan $PM/yolox_fp16.plan --fp16
python3 ~/ldrq-v3/base/perception/r1_perception_cli.py engine-check --engine $PM/yolox_fp16.plan --gate detections
```
最后一行必须是 `PARITY PASS`。真机用的是 FP16：在 TensorRT 8.5 上，FP32 的原始数值差一点没过门槛，但检测结果完全一样。

### A6. 安装 V3（约 5 分钟）

**开一个新终端**（终端里 source 过 `env.sh` 的话，安装程序会拒绝运行）：
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
bash ~/ldrq-v3/v3_package/install_v3.sh --src ~/ldrq-v3/base --dst ~/r1-robot-v3
```
每一步会打印 `[n/8]`，日志写在 `~/orin_commissioning/` 下。

| 步骤 | 应该看到 |
|---|---|
| 1 预检 | 磁盘空间够；包内文件与 `MANIFEST.sha256` 一致 |
| 2 对比 | 仅供参考：`bridge.yaml` 与 v2.0.0 不同（A4 写进去的 v2 范围）；如果还列出别的文件，说明底座里有人手动改过 |
| 3 复制 | 底座复制过去；`llm/ollama`、`llm/models` 以链接形式带过去（如果做了 A3） |
| 4 叠加 | `joint_map.hpp identical (v2's gains)`；`v3_models/: dance_wave` |
| 5 配置 | 只有预期的差异：`reflex: on`、`dance*`、`obj_stop_min_m 0.9`；`proximity_*`、`person_box_*`；`reflex_*` 一整段。**没有其他以 `<` 开头的行** |
| 6 编译 | 硬件桥和策略节点编译成功 |
| 7 检查 | `CycloneDDS from /usr/local/lib: OK`；V3 的每一行测试都是 `ok` |
| 8 引擎 | v2 引擎和舞蹈引擎各一行 `PARITY PASS`；**记下舞蹈引擎的 `fnv1a=`** |
| 结束 | `source tree unchanged: OK`；`V3 installed: .../r1-robot-v3` |

停在任何一步：把整段输出带回，什么都不要改。`~/ldrq-v3/base` 只被读取，不会被改动；要回到 v2，source 它的 `env.sh` 即可。

## B 部分：每次上机

### B1. 机器人动之前

1. 用手持遥控**切到开发者模式**。不切的话，宇树的运控也会给电机发指令，现象像是策略不好。
2. 每次开机后**锁频**一次（需要 sudo）：`bash ~/r1-robot-v3/deploy/tools/w08_preflight.sh --lock-clocks`，必须报告 0 FAIL。
3. **每次开机后都要打开相机视频流**：在宇树 APP 里**关 video_hub**、**开 Stereo patch PC1**。
4. 机器人吊上吊架，脚着地，绳子松。这时相机离地约 1.1–1.2 米。

**每个**终端都先执行下面这段：
```bash
export ROS_DOMAIN_ID=99 ROS_LOCALHOST_ONLY=1
source ~/r1-robot-v3/deploy/env.sh
R=~/r1-robot-v3; M=$R/mission_ctl/r1_mission_cli.py; PC=$R/perception/r1_perception_cli.py
PM=$R/perception/models; O=~/orin_commissioning; D=$R/v3_models/dance_wave; mkdir -p $O
```

**起感知之前先查相机画面。** 10-24 有一次上机在这里耽误了一个小时：感知 0 帧，硬件桥 `DEGRADED`。
```bash
pgrep -af stereo_capture                                   # 应该什么都不打印；有的话 kill <PID>
python3 $R/perception/tools/stereo_capture.py --probe      # 3 秒；5002、5003 两个端口都应约 10 fps
```
如果显示 0 个包：在 APP 里重新设置（关 video_hub、开 Stereo patch PC1），再 probe 一次。

### B2. 起服务

终端 1，模型（选了方式 C 就跳过）：
```bash
bash $R/llm/ollama_ctl.sh start && bash $R/llm/ollama_ctl.sh health
```
终端 2，感知：
```bash
cd $R && python3 $PC run --engine $PM/yolox_fp16.plan 2>&1 | tee $O/v3_perception.log
```
终端 3，栈，**输出关**：
```bash
PYTHONUNBUFFERED=1 ros2 launch r1_hw_bridge r1_stack.launch.py \
    engine:=$R1_DEPLOY_ROOT/artifacts/policy_fp32.plan iface:=eth10 \
    enable_output:=false kp_scale:=1.3 kd_scale:=1.0 \
    reflex_enable:=true dance_enable:=true \
    dance_engine:=$D/policy_fp32.plan dance_reference:=$D/dance_ref.bin \
    reflex_log:=$O/v3_reflex_log.csv 2>&1 | tee $O/v3_stack.log
```
- 状态行必须是 `RUNNING`。
- 启动时打印的 `dance engine ... fnv1a=...` 必须和 A6 记下的一致。
- 数值参数要带小数点（写 `1.3`，不要写 `1`）。
- `iface` 是机器人 `192.168.123.x` 网段的网卡名，实验室那台是 `eth10`。

如果状态行是 `DEGRADED`：
1. 先保存原因：`grep "DEGRADED:" $O/v3_stack.log`；
2. 在终端 3 按 Ctrl-C，重新起栈。输出关时重启没有风险；不要用 `ros2` 命令行发 resume；
3. 重启后还是 DEGRADED：停下，把日志带回。

终端 4，只看关键行：
```bash
tail -f $O/v3_stack.log | grep --line-buffered -E "DANCE|REFLEX:|DEGRADED"
```

### B3. 人距标定（可选）

`perception/config/perception.yaml` 里的 `person_box_k: 0.465` 是 2026-10-24 在实验室那台机器人吊装时标定的。如果你的相机高度不同，就重新标定一次。不需要卷尺：
1. 帮手面对机器人，伸直手臂，指尖刚好碰到机器人胸口：约 0.65 米；
2. 帮手再正常后退一步：约 1.3 米。

```bash
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 0.65 --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --seconds 5 --distance 1.3  --csv $O/v3_cal.csv
python3 $R/perception/tools/person_box_cal.py --from-csv $O/v3_cal.csv --write
```
`--write` 会把旧文件保留为 `perception.yaml.before_cal`。标定后要重启感知（终端 2）。

## C 部分：输出关的检查（约 40 分钟）

电机收不到任何指令，但两个策略、conductor 和反射都照常运行。每项检查都看终端 4 和 `ask` 的回复。

| # | 做什么 | 通过 |
|---|---|---|
| ① | 前方 2 米内没人：`python3 $M ask "dance for 10 seconds"`，按 `y` | 终端 4 依次出现 `walk -> dance, DANCING`、`ENDING`、`dance -> walk, OFF (danced; back to standing)`；回复是 `1. Danced on the spot for 1x.x s.` 和 `Done.`；没有 Traceback |
| ② | 做 3 次，每次间隔 15 秒以上：`ask "dance for 30 seconds"`。帮手一开始站在正前方约 2.5 米，跳起来以后**从正前方**慢慢走到一臂远 | 同一时刻出现 `DANCE: policy dance -> walk ... stepped back` 和 `REFLEX: stepping back: person too close -- person 0.xx m`；回复是 `Interrupted: stepped back because of a person ...` |
| ②b | 帮手站在一臂远不动：`ask "dance"` | 被拒：`1. Did not dance. Interrupted: the bridge did not dance: ...too close` |
| ③ | `ask "dance for 30 seconds"`，跳起来几秒后在 `ask` 的终端按 Ctrl-C | 出现 `ENDING`，3 秒内出现 `dance -> walk`；回复里有 `Interrupted: operator interrupt (Ctrl-C)` |
| ④ | （只记录）帮手先站在 1.0–1.1 米不动，再 `ask "dance for 10 seconds"`，做 5 次 | 每次记下三种结果之一：被拒 / 跳了但 2 秒内就后退 / 跳完了 |

②每次都要记下：`REFLEX` 那一行里的距离，以及帮手是不是从正前方过来的。

**通过条件**（通过了才能做 D 部分）：
- ① 正确；
- ② 3 次都是同一时刻交接，回复正确；
- ③ 正确；
- 全程没有 Traceback。

## D 部分：吊装、电机输出开（约 60 分钟）

C 部分通过后才做。吊着时脚上受力不足，跳舞看起来会和落地时不同。**这一部分只看安全**，不评动作好坏：
- 不摔；
- 交接时腿不乱蹬；
- 绳子不会突然吃力。

终端 3：先按 Ctrl-C 停掉输出关的栈，再改成 `enable_output:=true` 起栈（日志名改为 `v3m_...`）。等到 `RUNNING`，看机器人用 v2 站稳 10 秒，没有异常再往下。

| # | 做什么 | 通过 |
|---|---|---|
| ① | `ask "dance for 30 seconds"`，做 3 次，每次之间站 15 秒以上 | 每次 30–33 秒后回到站立；没有脚滑、腿软，绳子没有突然吃力 |
| ② | 跳舞中途用假相机给一个"人在 0.5 米"，做 10 次（方法见下） | 每次都在那个时刻切回 v2，后退约 1.5 秒后站住；0 次异常 |
| ③ | 帮手从正前方走近，做 5 次，方法同 C 部分 ② | 同 ②；回复里有 `stepped back because of a person` |
| ④ | 跳舞时按 Ctrl-C，做 2 次 | 出现 `ENDING`，3 秒内在小节结尾回到站立 |

**② 不需要真人。** 先在终端 2 按 Ctrl-C 停掉感知：两个程序同时发近距话题会互相干扰。然后每一次都这样做：
1. 在终端 5 起假相机：
```bash
python3 $R/v3_tools/dance_devbox_camera.py --seconds 45 --timeline "0:none,12:0.5,14:none"
```
2. **5 秒内**在终端 6 发指令（`run` 不经过模型，也不用按 `y`）：
```bash
python3 $M run "dance 30s" --yes
```
3. 到第 12 秒，假相机打印 `person 0.50 m`。这时终端 4 应同时出现交接和 `REFLEX` 两行，机器人后退约 1.5 秒后站住。
4. 等假相机的 45 秒结束，再做下一次。

10 次里人出现的时间分别用第 12、15、18、21、24、13、16、19、22、25 秒，让它落在舞蹈的不同位置（例如第 2 次是 `"0:none,15:0.5,17:none"`）。

假相机停了以后硬件桥收不到画面，会保持站立，这是正常的。10 次做完后，在终端 2 重新起感知。

## 停止方式和各状态的含义

| 操作 / 情况 | 结果 |
|---|---|
| 在 `ask` / `run` 的终端按 **Ctrl-C** | 走路时：立即发零速度。跳舞时：`ENDING`，在小节结尾交回 v2（3 秒内） |
| `ask` 进程意外退出 | 跳舞心跳中断（0.5 秒）：在小节结尾结束跳舞；走路时速度指令在 500 ms 内降到零 |
| 有人进入触发距离 | RETREAT：v2 以 −0.3 m/s 后退 1.5 秒，然后 STAND，直到操作员确认新计划、且前方没人 |
| 没有相机画面（超过 0.3 秒） | 禁止向前走；不会开始跳舞，正在跳的会在小节结尾结束 |
| 硬件桥检测到故障 | **DEGRADED**：电机转为阻尼。如果电机输出开着：急停，记下现象，本次上机到此为止 |
| 任何意外情况 | **急停** |

收尾：确认机器人吊着、静止，然后依次在终端 3 按 Ctrl-C 停栈、在终端 2 停感知，最后执行 `bash $R/llm/ollama_ctl.sh stop`。

## 故障排查

| 现象 | 原因和处理 |
|---|---|
| `install_v3.sh` 第 1 步报 "this terminal has sourced a tree's env.sh" | 开一个新终端，运行安装程序前什么都不要 source |
| 第 1 步报 "has no perception/" | `--src` 必须是 `~/ldrq-v3/base` |
| 第 4 步报 `joint_map.hpp ... differs` | 底座没装 v2 策略：做 A4 |
| 第 5 步出现别的 `<` 行 | 底座配置里有本地改动：把它抄进 `~/r1-robot-v3` 里的新文件 |
| 第 8 步 `PARITY FAILED` | 不要运行这个引擎，把日志带回 |
| 感知 0 帧，随后硬件桥 `DEGRADED` | 检查 APP 设置（每次开机都会恢复默认），或者 `stereo_capture.py` 还占着 5002/5003 |
| `ask "dance"` 被拒："no status from the bridge on /r1_hw_bridge/reflex" | 起栈时没加 `reflex_enable:=true`（跳舞必须开反射） |
| 一直不开始跳舞 | 相机视野里 1 米内有人或东西，或者机器人没站稳满 1 秒 |
| 刚起栈就 `DEGRADED: joint_target stale`（输出关时出现过一次） | 重启栈；之前重启后没再出现。如果又出现，把日志带回 |
| 硬件桥启动就退出：`InvalidParameterTypeException` | 某个数值启动参数写成了整数 |
| 指令没有效果 | 各终端的 `ROS_DOMAIN_ID` / `ROS_LOCALHOST_ONLY` 不一致，或没有 source `env.sh` |
| `ros2 topic list` 崩溃（`bad_alloc`） | 这台机器人上的 DDS 库冲突，是已知问题；除了 `ros2 launch`，本系统不需要 `ros2` 命令行 |

## 回到 v2、卸载 V3

- **回到 v2**：改为 `source ~/ldrq-v3/base/deploy/env.sh`。底座没有被改动过。在那里起栈时不加 `reflex_*` / `dance_*` 参数。
- **卸载 V3**：`rm -rf -- ~/r1-robot-v3`，底座仍然可用。
- **全部删除**：先停掉模型服务，再 `rm -rf -- ~/ldrq-v3 ~/r1-robot-v3`。如果用了方式 A，被链接的 release 目录保持原样。

## 带回什么

```bash
cd ~ && tar czf v3_logs_$(date +%m%d).tar.gz orin_commissioning/v3*
```
另附每项检查的记录：结果、触发距离、有什么异常、大概在什么时间。录像如果要公开，不能有人脸，并去掉音频；拍到人的双目录制数据不公开。
