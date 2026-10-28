// R1 hardware bridge: robot DDS <-> generic ROS 2 topics.
//
//   rt/lowstate (DDS)  ->  ~/obs            85 floats, one frame per control step
//   ~/joint_target     ->  rt/lowcmd (DDS)  24 targets -> 35 motor slots
//
// The policy node speaks Float32MultiArray and knows nothing of Unitree, which
// is what lets it be unit-tested, replayed from a rosbag and driven from
// simulation with no SDK present.
//
// ⚠️ This translation unit must NEVER include a Unitree header. The SDK ships
// CycloneDDS 0.10.2 under /usr/local/include/dds/, ROS foxy ships 0.7.0 under
// /opt/ros/foxy/include/dds/, and a target that sees both compiles against a
// blend of the two. All robot I/O goes through robot_io.hpp, whose
// implementation is compiled without ROS on the include path.
//
// Two rates, on purpose:
//   * observation/policy at control_rate_hz (50) -- the rate the policy trained at
//   * command at cmd_rate_hz (500) -- what the motor bus expects; the per-joint
//     PD loop itself runs in the motor firmware, so nothing here integrates.
// Targets are re-sent unchanged between policy ticks rather than interpolated:
// in training the target was a 20 ms step, and smoothing it here would be a
// sim-to-real difference introduced by the deployment code.
//
// SAFETY: `enable_output` defaults to FALSE. Nothing is written to rt/lowcmd
// until it is explicitly turned on, so building and running this node cannot
// move the robot by accident.
//
// V3: the proximity reflex (reflex_fsm.hpp, `reflex_enable`, default FALSE). Once
// per control step it sits between ~/cmd_vel and the observation's velocity
// command: forward is blocked inside a speed-dependent stop distance, a person or
// obstacle too close starts a short retreat, and then the robot stands until
// ~/reflex_reset. It reads /r1_perception/proximity. With it disabled none of
// its topics exist and the observation is built exactly as before.
//
// V3 dance (`dance_enable`, default FALSE; needs the reflex): two policies run in the
// runner (its fork in v3/ros2_ws) -- v2 and the dance policy -- and this node chooses
// which reaches the motors, through the conductor (conductor.hpp, the reference is
// v3/reflex/conductor.py): ~/dance (Bool, from mission_ctl: true = dance and heartbeat,
// false = end) starts and ends a dance; the reflex ends it at once and v2 steps back.
// The observation then carries 6 more floats for the runner (IMU quaternion w x y z,
// dance tick, policy), and its last-action slot is the action that was applied, from
// whichever policy that was.

#include "r1_hw_bridge/joint_map.hpp"
#include "r1_hw_bridge/conductor.hpp"
#include "r1_hw_bridge/reflex_fsm.hpp"
#include "r1_hw_bridge/robot_io.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace r1_hw_bridge
{

using std::chrono::steady_clock;
using std::chrono::duration;

static_assert(kNumSlots == kSlots, "joint_map.hpp and robot_io.hpp disagree on slot count");

enum class State { WaitingState, WaitingPolicy, Running, Degraded };

const char * StateName(State s)
{
  switch (s) {
    case State::WaitingState: return "WAITING_LOWSTATE";
    case State::WaitingPolicy: return "WAITING_POLICY";
    case State::Running: return "RUNNING";
    case State::Degraded: return "DEGRADED";
  }
  return "?";
}

// Isaac Lab's quat_rotate_inverse for q = (w, x, y, z). The component order was
// established on the robot (probe_lowstate: upright reads (0,0,-1) only under
// this ordering, and the accelerometer agrees to 0.04 m/s^2). Do not "fix" it.
inline void QuatRotateInverse(const std::array<float, 4> & q, const float v[3], float out[3])
{
  const float w = q[0];
  const float vec[3] = {q[1], q[2], q[3]};
  const float dot = vec[0] * v[0] + vec[1] * v[1] + vec[2] * v[2];
  const float cross[3] = {vec[1] * v[2] - vec[2] * v[1],
                          vec[2] * v[0] - vec[0] * v[2],
                          vec[0] * v[1] - vec[1] * v[0]};
  for (int i = 0; i < 3; ++i) {
    out[i] = v[i] * (2.0f * w * w - 1.0f) - cross[i] * w * 2.0f + vec[i] * dot * 2.0f;
  }
}

class BridgeNode : public rclcpp::Node
{
public:
  BridgeNode()
  : rclcpp::Node("r1_hw_bridge")
  {
    RobotIoConfig io_cfg;
    io_cfg.iface = declare_parameter<std::string>("iface", "eth10");
    io_cfg.domain = static_cast<int>(declare_parameter<int64_t>("dds_domain", 0));
    io_cfg.state_topic = declare_parameter<std::string>("lowstate_topic", "rt/lowstate");
    io_cfg.cmd_topic = declare_parameter<std::string>("lowcmd_topic", "rt/lowcmd");
    io_cfg.check_state_crc = declare_parameter<bool>("check_state_crc", true);
    io_cfg.enable_output = declare_parameter<bool>("enable_output", false);

    control_rate_hz_ = declare_parameter<double>("control_rate_hz", 50.0);
    cmd_rate_hz_ = declare_parameter<double>("cmd_rate_hz", 500.0);
    // Gains are PER JOINT and come from the training config via kKp/kKd; these
    // two only scale them. The first version of this node had one global kp/kd
    // for all 24 joints, which is not a simplification of six actuator groups
    // spanning kp 20..100 -- it is a different controller. Measured on the robot
    // 2026-09-04: legs with no perceptible damping (their group wants kp 100,
    // kd 2, they got 10 and 1) and a head oscillating at high frequency (its
    // group wants kd 1, it got 3).
    //
    // Scale 1.0 IS the trained controller. Ramp with kp_scale, not with an
    // absolute number, so the endpoint of the ramp is known rather than guessed.
    kp_scale_ = declare_parameter<double>("kp_scale", 1.0);
    kd_scale_ = declare_parameter<double>("kd_scale", 1.0);
    // The head's encoder zero is not the URDF zero. Measured 2026-10-28: with the
    // head facing straight ahead (as the factory controller leaves it), head_yaw
    // reads +0.752 rad, so holding it at the URDF default turned it ~43 deg to the
    // right on every start. These are the readings at the URDF zero; they shift
    // both the hold target and the observation, so the policy still sees the
    // head where training had it.
    joint_zero_.fill(0.0f);
    joint_zero_[kHeadPitch] = static_cast<float>(declare_parameter<double>("head_pitch_zero", 0.0));
    joint_zero_[kHeadYaw] = static_cast<float>(declare_parameter<double>("head_yaw_zero", 0.0));
    state_timeout_ms_ = declare_parameter<double>("lowstate_timeout_ms", 100.0);
    target_timeout_ms_ = declare_parameter<double>("target_timeout_ms", 60.0);
    min_rate_hz_ = declare_parameter<double>("min_control_rate_hz", 45.0);
    auto_recover_ = declare_parameter<bool>("auto_recover", false);
    cmd_vel_timeout_ms_ = declare_parameter<double>("cmd_vel_timeout_ms", 500.0);
    // The command envelope the policy was actually TRAINED on -- the end point of
    // W04's command curriculum (COMMAND_RANGES_FINAL in
    // tasks/r1_flat/flat_env_cfg.py), not a comfort limit. Outside it the policy
    // is extrapolating, and a biped extrapolating is a fall.
    //
    // Note the y range: lin_vel_y was pinned to 0 for the WHOLE of training, so
    // sideways is not "weakly trained", it is never-seen. It gets its own message
    // below rather than being lumped in with a clamp.
    //
    // Overridable, because an out-of-distribution command is a legitimate
    // experiment -- but it has to be asked for, not arrived at by typing a number.
    cmd_vel_x_range_ = declare_parameter<std::vector<double>>(
      "cmd_vel_x_range", {0.0, 1.0});
    cmd_vel_y_range_ = declare_parameter<std::vector<double>>(
      "cmd_vel_y_range", {0.0, 0.0});
    cmd_vel_yaw_range_ = declare_parameter<std::vector<double>>(
      "cmd_vel_yaw_range", {-0.5, 0.5});
    for (auto * r : {&cmd_vel_x_range_, &cmd_vel_y_range_, &cmd_vel_yaw_range_}) {
      if (r->size() != 2 || (*r)[0] > (*r)[1]) {
        throw std::invalid_argument(
          "cmd_vel_*_range must be [min, max] with min <= max");
      }
    }
    RCLCPP_INFO(get_logger(),
      "cmd_vel envelope (as trained): vx [%.2f, %.2f]  vy [%.2f, %.2f]  wz [%.2f, %.2f]",
      cmd_vel_x_range_[0], cmd_vel_x_range_[1], cmd_vel_y_range_[0],
      cmd_vel_y_range_[1], cmd_vel_yaw_range_[0], cmd_vel_yaw_range_[1]);
    DeclareReflex();

    const auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort();
    obs_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/obs", qos);
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "~/status", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local());
    reset_pub_ = create_publisher<std_msgs::msg::Empty>(
      "~/policy_reset", rclcpp::QoS(rclcpp::KeepLast(1)).reliable());
    // Raw state, republished for rosbag and `ros2 topic echo`. This is the
    // observability that made unitree_ros2 look attractive; publishing it here
    // gets the same thing without a second DDS stack.
    dbg_jpos_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_pos", qos);
    dbg_jvel_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_vel", qos);
    dbg_imu_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/imu", qos);
    dbg_tau_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_tau", qos);
    dbg_cmd_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/cmd_debug", qos);

    target_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "joint_target", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnTarget(*m);});
    action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "action", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnAction(*m);});
    cmdvel_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "~/cmd_vel", qos,
      [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnCmdVel(*m);});
    resume_sub_ = create_subscription<std_msgs::msg::Bool>(
      "~/resume", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      [this](std_msgs::msg::Bool::ConstSharedPtr m) {OnResume(*m);});
    if (HaveReflex()) {
      // BEST_EFFORT reads perception's RELIABLE publisher; freshness is judged on
      // the frame's capture time, not on delivery.
      prox_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
        reflex_topic_, rclcpp::QoS(rclcpp::KeepLast(1)).best_effort(),
        [this](std_msgs::msg::Float64MultiArray::ConstSharedPtr m) {OnProximity(*m);});
      reflex_reset_sub_ = create_subscription<std_msgs::msg::Empty>(
        "~/reflex_reset", rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
        [this](std_msgs::msg::Empty::ConstSharedPtr) {
          std::lock_guard<std::mutex> lock(reflex_mutex_);
          reflex_reset_pending_ = true;
        });
      // RELIABLE and a little history: mission_ctl must not miss the one message
      // that says a retreat started.
      reflex_pub_ = create_publisher<std_msgs::msg::String>(
        "~/reflex", rclcpp::QoS(rclcpp::KeepLast(20)).reliable());
    }
    if (conductor_) {
      dance_sub_ = create_subscription<std_msgs::msg::Bool>(
        "~/dance", rclcpp::QoS(rclcpp::KeepLast(5)).reliable(),
        [this](std_msgs::msg::Bool::ConstSharedPtr m) {
          std::lock_guard<std::mutex> lock(reflex_mutex_);
          // an end request is never overwritten by a heartbeat in the same step
          if (!m->data) {dance_pending_ = reflex::DanceMsg::End;}
          else if (dance_pending_ != reflex::DanceMsg::End) {dance_pending_ = reflex::DanceMsg::Dance;}
        });
      dance_target_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "dance_joint_target", qos,
        [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {OnDanceTarget(*m);});
      dance_action_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "dance_action", qos,
        [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr m) {
          if (m->data.size() != kNumActions) {return;}
          std::lock_guard<std::mutex> lock(target_mutex_);
          std::copy(m->data.begin(), m->data.end(), dance_last_action_.begin());
        });
    }

    frame_.assign(kFrameDim, 0.0f);
    last_action_.assign(kNumActions, 0.0f);
    target_.assign(kNumActions, 0.0f);
    dance_last_action_.assign(kNumActions, 0.0f);
    dance_target_.assign(kNumActions, 0.0f);
    cmd_vel_ = {0.0f, 0.0f, 0.0f};

    io_ = std::make_unique<RobotIo>(io_cfg);
    io_->Start([this](const RobotState & s) {OnRobotState(s);});

    if (io_->output_enabled()) {
      RCLCPP_WARN(get_logger(), "enable_output=true -- THIS NODE WILL COMMAND THE ROBOT");
    } else {
      RCLCPP_INFO(get_logger(),
                  "enable_output=false -- commands computed and published to "
                  "~/cmd_debug only, nothing written to %s", io_cfg.cmd_topic.c_str());
    }

    obs_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / control_rate_hz_), [this]() {OnObsTick();});
    status_timer_ = create_wall_timer(std::chrono::seconds(1), [this]() {OnStatusTick();});
    cmd_thread_ = std::thread([this]() {CmdLoop();});

    RCLCPP_INFO(get_logger(),
                "bridge up: iface=%s domain=%d obs %.0f Hz, cmd %.0f Hz, %zu joints",
                io_cfg.iface.c_str(), io_cfg.domain, control_rate_hz_, cmd_rate_hz_,
                kNumJoints);

    // The robot link above and the ROS side below are separate DDS participants.
    // Print the ROS side's isolation settings: joining the factory stack's
    // domain kills discovery with std::bad_alloc, and the symptom ("topic does
    // not appear to be published") points at the publisher rather than at the
    // domain. Having it in the log makes that diagnosis a one-line check.
    const char * dom = std::getenv("ROS_DOMAIN_ID");
    const char * loc = std::getenv("ROS_LOCALHOST_ONLY");
    // Print the gains that will actually be sent. A scale is easy to pass and
    // easy to forget; the numbers below are what reaches the motors.
    RCLCPP_INFO(get_logger(),
                "gains: kp_scale=%.3f kd_scale=%.3f -> hip kp=%.1f kd=%.1f | "
                "ankle kp=%.1f | head kp=%.1f kd=%.1f  (1.0 = as trained)",
                kp_scale_, kd_scale_,
                kKp[0] * kp_scale_, kKd[0] * kd_scale_,
                kKp[16] * kp_scale_, kKp[8] * kp_scale_, kKd[8] * kd_scale_);
    RCLCPP_INFO(get_logger(), "head zero (encoder reading at the URDF zero): pitch %+.4f yaw %+.4f rad",
                joint_zero_[kHeadPitch], joint_zero_[kHeadYaw]);
    if (kp_scale_ > 1.0 || kd_scale_ > 1.0) {
      RCLCPP_WARN(get_logger(),
                  "kp_scale/kd_scale above 1.0 exceeds the trained gains -- the "
                  "policy has never seen this plant");
    }

    RCLCPP_INFO(get_logger(), "ROS side: ROS_DOMAIN_ID=%s ROS_LOCALHOST_ONLY=%s rmw=%s",
                dom ? dom : "(unset -> 0)", loc ? loc : "(unset -> 0)",
                rmw_get_implementation_identifier());
    if (!dom || !loc || std::string(loc) != "1") {
      RCLCPP_WARN(get_logger(),
        "ROS side is NOT isolated. On this robot that usually means other ROS "
        "processes will die in discovery (std::bad_alloc) and see this node's "
        "topics as unpublished. `source env.sh` sets ROS_DOMAIN_ID=99 and "
        "ROS_LOCALHOST_ONLY=1 -- every terminal needs the same values.");
    }
  }

  ~BridgeNode() override
  {
    running_.store(false);
    if (cmd_thread_.joinable()) {cmd_thread_.join();}
  }

private:
  static constexpr std::size_t kFrameDim = 3 + 3 + 3 + kNumJoints + kNumJoints + kNumActions;

  // The generated header comes from whichever training run dump_interface.py
  // was pointed at. Regenerating it from a run with a different action space
  // would silently change the frame width; the policy node would then reject
  // every frame at runtime. Fail at compile time instead.
  static_assert(kNumJoints == 26, "joint_map.hpp regenerated from a different run");
  static_assert(kNumActions == 24, "joint_map.hpp regenerated from a different run");
  static_assert(kFrameDim == 85, "frame width no longer matches the policy");
  // The two joints the policy does not drive (test_joint_map checks which).
  static constexpr std::size_t kHeadPitch = 8, kHeadYaw = 13;

  // ---------------------------------------------------------- robot ingress
  void OnRobotState(const RobotState & s)
  {
    std::lock_guard<std::mutex> lock(state_mutex_);
    latest_ = s;
    have_state_ = true;
    last_state_time_ = steady_clock::now();
  }

  // ------------------------------------------------------------ ROS ingress
  void OnTarget(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != kNumActions) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
                            "joint_target has %zu values, expected %zu",
                            m.data.size(), kNumActions);
      return;
    }
    for (const float v : m.data) {
      if (!std::isfinite(v)) {
        RCLCPP_ERROR(get_logger(), "joint_target contains NaN/inf -- degrading");
        Degrade("non-finite joint_target");
        return;
      }
    }
    const auto now = steady_clock::now();
    RecordLoopLag(now);
    std::lock_guard<std::mutex> lock(target_mutex_);
    std::copy(m.data.begin(), m.data.end(), target_.begin());
    last_target_time_ = now;
    have_target_ = true;
  }

  // Setpoint lag: from publishing an observation to having the setpoint it
  // produced. This is exactly the quantity training randomized -- W04's
  // DelayedJointPositionAction lags the setpoint by {0,1} control steps, i.e.
  // {0, 20} ms at 50 Hz (tasks/r1_flat/mdp.py) -- so it is the number to
  // compare against, and it is directly measurable here rather than inferred
  // by cross-correlating command against motion. The actuator's own response
  // lag is a separate quantity and is NOT what that randomization models.
  //
  // Assumes the policy emits one target per obs frame, true in steady state at
  // 50 Hz. A dropped frame shows up as a sample one control period too long,
  // which is visible as a bimodal spread rather than a quietly wrong median.
  void RecordLoopLag(steady_clock::time_point now)
  {
    const int64_t pub_ns = obs_pub_ns_.load();
    if (pub_ns == 0) {return;}
    const double ms =
      duration<double, std::milli>(now - steady_clock::time_point(steady_clock::duration(pub_ns)))
      .count();
    // Negative or absurd means the pairing is not 1:1 (startup, or a burst);
    // drop rather than poison the statistic.
    if (ms < 0.0 || ms > 1000.0) {return;}
    std::lock_guard<std::mutex> lock(lag_mutex_);
    if (lag_samples_.size() < kLagSamples) {
      lag_samples_.push_back(ms);
    } else {
      lag_samples_[lag_next_] = ms;
      lag_next_ = (lag_next_ + 1) % kLagSamples;
    }
  }

  // p50/p95/max over the window, or all zeros if nothing has been paired yet.
  std::array<double, 3> LoopLag()
  {
    std::vector<double> v;
    {
      std::lock_guard<std::mutex> lock(lag_mutex_);
      v = lag_samples_;
    }
    if (v.empty()) {return {0.0, 0.0, 0.0};}
    std::sort(v.begin(), v.end());
    auto at = [&v](double q) {
        return v[std::min(v.size() - 1,
          static_cast<std::size_t>(q * static_cast<double>(v.size())))];
      };
    return {at(0.50), at(0.95), v.back()};
  }

  void OnAction(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != kNumActions) {return;}
    std::lock_guard<std::mutex> lock(target_mutex_);
    std::copy(m.data.begin(), m.data.end(), last_action_.begin());
  }

  void OnCmdVel(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != 3) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "~/cmd_vel expects 3 floats [vx, vy, wz]");
      return;
    }

    // Clamp to the trained envelope. This command goes STRAIGHT into the
    // velocity_commands observation term -- there is nothing downstream that
    // checks it, and the policy has no way to signal "I was not trained for
    // this". It just produces confident joint targets for a state it has never
    // seen. So the check belongs here, and it has to be loud: a silently
    // accepted out-of-range command is the kind of thing that reads as a
    // sim2real gap afterwards.
    const std::array<const std::vector<double> *, 3> ranges{
      &cmd_vel_x_range_, &cmd_vel_y_range_, &cmd_vel_yaw_range_};
    static constexpr const char * kAxis[3] = {"vx", "vy", "wz"};
    std::array<float, 3> want{m.data[0], m.data[1], m.data[2]};
    std::array<float, 3> got = want;
    for (int i = 0; i < 3; ++i) {
      const auto lo = static_cast<float>((*ranges[i])[0]);
      const auto hi = static_cast<float>((*ranges[i])[1]);
      got[i] = std::min(hi, std::max(lo, want[i]));
      if (got[i] == want[i]) {continue;}
      if (lo == 0.0f && hi == 0.0f) {
        // Not "outside the range" -- this axis was pinned to zero for the whole
        // of training, so there is no range to be outside of.
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "~/cmd_vel %s=%.3f ignored: this axis was pinned to 0 for ALL of "
          "training, so the policy has never seen a non-zero value. Commanding "
          "it is not a small extrapolation, it is a state that does not exist in "
          "the training distribution. Override with the %s range parameter only "
          "if that is the experiment you mean to run.",
          kAxis[i], static_cast<double>(want[i]), kAxis[i]);
      } else {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          "~/cmd_vel %s=%.3f clamped to %.3f -- trained range is [%.2f, %.2f]",
          kAxis[i], static_cast<double>(want[i]), static_cast<double>(got[i]),
          static_cast<double>(lo), static_cast<double>(hi));
      }
    }

    std::lock_guard<std::mutex> lock(target_mutex_);
    cmd_vel_ = {got[0], got[1], got[2]};
    last_cmdvel_time_ = steady_clock::now();
    have_cmdvel_ = true;
  }

  void OnResume(const std_msgs::msg::Bool & m)
  {
    if (!m.data) {return;}
    if (state_.load() != State::Degraded) {
      RCLCPP_INFO(get_logger(), "~/resume ignored: not degraded (%s)",
                  StateName(state_.load()));
      return;
    }
    // Recovery is explicit and always re-warms: the policy's 5-frame history now
    // straddles a gap, which is out of the training distribution. Clearing it and
    // refilling is the only correct restart.
    RCLCPP_WARN(get_logger(), "resuming: clearing policy history, re-warming");
    reset_pub_->publish(std_msgs::msg::Empty());
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      have_target_ = false;
    }
    state_.store(State::WaitingPolicy);
    PublishStatus();
  }

  // ------------------------------------------------------------ observation
  void OnObsTick()
  {
    const auto now = steady_clock::now();
    RobotState s;
    bool ok;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      ok = have_state_ &&
        duration<double, std::milli>(now - last_state_time_).count() < state_timeout_ms_;
      s = latest_;
    }
    if (!ok) {
      if (state_.load() != State::Degraded && state_.load() != State::WaitingState) {
        Degrade("lowstate stale");
      }
      return;
    }
    if (state_.load() == State::WaitingState) {
      state_.store(State::WaitingPolicy);
      PublishStatus();
    }

    const float down[3] = {0.0f, 0.0f, -1.0f};
    float grav[3];
    QuatRotateInverse(s.imu.quat, down, grav);

    std::size_t k = 0;
    for (int i = 0; i < 3; ++i) {frame_[k++] = s.imu.gyro[i];}
    for (int i = 0; i < 3; ++i) {frame_[k++] = grav[i];}
    {
      // A teleop that dies must not leave the robot walking on its last command.
      std::array<float, 3> cmd{};
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        const bool cmd_fresh = have_cmdvel_ &&
          duration<double, std::milli>(now - last_cmdvel_time_).count() < cmd_vel_timeout_ms_;
        if (!cmd_fresh && have_cmdvel_) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
            "~/cmd_vel stale -- commanding zero velocity");
        }
        for (int i = 0; i < 3; ++i) {cmd[i] = cmd_fresh ? cmd_vel_[i] : 0.0f;}
      }
      if (HaveReflex()) {ApplyReflex(now, cmd);}
      for (int i = 0; i < 3; ++i) {frame_[k++] = cmd[i];}
    }
    // joint_pos is RELATIVE to the default pose; joint_vel is absolute. Getting
    // this backwards produces a plausible-looking observation and a policy that
    // has never seen anything like it.
    const std::size_t jpos0 = k;
    for (std::size_t j = 0; j < kNumJoints; ++j) {
      frame_[k++] = s.motor[kJointSlot[j]].q - joint_zero_[j] - kDefaultPos[j];
    }
    const std::size_t jvel0 = k;
    for (std::size_t j = 0; j < kNumJoints; ++j) {
      frame_[k++] = s.motor[kJointSlot[j]].dq;
    }
    {
      // The action applied over the last control step: from the policy that was
      // driving then (V3: v2's, or the dance's).
      std::lock_guard<std::mutex> lock(target_mutex_);
      const auto & applied = applied_policy_ == reflex::Policy::Dance ? dance_last_action_ : last_action_;
      for (std::size_t a = 0; a < kNumActions; ++a) {frame_[k++] = applied[a];}
    }
    applied_policy_ = policy_.load();

    obs_msg_.data.assign(frame_.begin(), frame_.end());
    if (conductor_) {
      // V3: what the runner needs for the dance policy (dance_obs.hpp)
      for (int i = 0; i < 4; ++i) {obs_msg_.data.push_back(s.imu.quat[i]);}
      obs_msg_.data.push_back(static_cast<float>(dance_tick_));
      obs_msg_.data.push_back(policy_.load() == reflex::Policy::Dance ? 1.0f : 0.0f);
    }
    obs_pub_->publish(obs_msg_);
    // Stamped so OnTarget can measure the loop lag this obs frame incurs. This
    // runs on the SDK's DDS thread while OnTarget runs on the ROS executor, so
    // the handoff is an atomic, not the target_mutex_.
    obs_pub_ns_.store(steady_clock::now().time_since_epoch().count());

    dbg_jpos_pub_->publish(Slice(jpos0, kNumJoints));
    dbg_jvel_pub_->publish(Slice(jvel0, kNumJoints));
    std_msgs::msg::Float32MultiArray imu_msg;
    imu_msg.data = {s.imu.quat[0], s.imu.quat[1], s.imu.quat[2], s.imu.quat[3],
                    s.imu.gyro[0], s.imu.gyro[1], s.imu.gyro[2],
                    grav[0], grav[1], grav[2]};
    dbg_imu_pub_->publish(imu_msg);

    // Measured joint torque, straight out of LowState's tau_est. It is not part
    // of the observation -- the policy never sees it -- but it is the only
    // DIRECT measure of how hard the hardware is working. W07 had to infer
    // torque as kKp * tracking_error to rule out saturation on the hip rolls,
    // and W08's INT8 acceptance gates on "torque stays inside its rating",
    // which should not be computed from the tracking error it is meant to
    // explain. Some firmware leaves tau_est at zero; a reader that sees all
    // zeros should say so rather than report 0 N*m as a measurement.
    std_msgs::msg::Float32MultiArray tau_msg;
    tau_msg.data.resize(kNumJoints);
    for (std::size_t j = 0; j < kNumJoints; ++j) {
      tau_msg.data[j] = s.motor[kJointSlot[j]].tau;
    }
    dbg_tau_pub_->publish(tau_msg);

    TrackRate(now);
  }

  std_msgs::msg::Float32MultiArray Slice(std::size_t off, std::size_t n) const
  {
    std_msgs::msg::Float32MultiArray m;
    m.data.assign(frame_.begin() + off, frame_.begin() + off + n);
    return m;
  }

  void TrackRate(steady_clock::time_point now)
  {
    // Start the first window at the first frame, not at construction. DDS
    // discovery takes ~100-400 ms, and counting that dead time as missed
    // frames made every single start-up log "observation rate 44.6 Hz below
    // 45.0 Hz (1/3)". A degrade counter that cries wolf on every boot is worse
    // than no counter: it trains you to read past the one that matters.
    if (!rate_primed_) {
      rate_primed_ = true;
      rate_window_ = now;
      obs_count_ = 0;
      return;
    }
    ++obs_count_;
    const double dt = duration<double>(now - rate_window_).count();
    if (dt < 1.0) {return;}
    const double hz = obs_count_ / dt;
    obs_count_ = 0;
    rate_window_ = now;
    obs_hz_.store(hz);
    if (hz < min_rate_hz_ && state_.load() == State::Running) {
      if (++slow_windows_ >= 3) {
        Degrade("control rate below threshold");
      } else {
        RCLCPP_WARN(get_logger(), "observation rate %.1f Hz below %.1f Hz (%d/3)",
                    hz, min_rate_hz_, slow_windows_);
      }
    } else if (hz >= min_rate_hz_) {
      slow_windows_ = 0;
    }
  }

  // ---------------------------------------------------------------- command
  std::string Reason()
  {
    std::lock_guard<std::mutex> lock(reason_mutex_);
    return degrade_reason_;
  }

  void Degrade(const std::string & why)
  {
    // exchange, not check-then-set: the observation timer and the command
    // thread can trip this in the same instant.
    if (state_.exchange(State::Degraded) == State::Degraded) {return;}
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      degrade_reason_ = why;
    }
    RCLCPP_ERROR(get_logger(), "DEGRADED: %s -- holding damping until ~/resume", why.c_str());
    PublishStatus();
  }

  // Runs at cmd_rate_hz on its own thread so a slow inference tick or a busy
  // ROS executor cannot stall the motor bus.
  void CmdLoop()
  {
    const auto period = std::chrono::duration_cast<steady_clock::duration>(
      std::chrono::duration<double>(1.0 / cmd_rate_hz_));
    auto next = steady_clock::now();

    while (running_.load()) {
      next += period;
      const auto now = steady_clock::now();

      bool state_fresh;
      RobotState s;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_fresh = have_state_ &&
          duration<double, std::milli>(now - last_state_time_).count() < state_timeout_ms_;
        s = latest_;
      }

      bool target_fresh = false;
      std::vector<float> target;
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_fresh = have_target_ &&
          duration<double, std::milli>(now - last_target_time_).count() < target_timeout_ms_;
        target = target_;
        // V3: the dance policy's target while it drives. Its freshness is the
        // conductor's business (it hands back to v2 within a control step); this
        // only refuses a target older than v2's own limit.
        if (policy_.load() == reflex::Policy::Dance && have_dance_target_ &&
          duration<double, std::milli>(now - last_dance_target_time_).count() < target_timeout_ms_)
        {
          target = dance_target_;
        }
      }

      if (state_.load() == State::WaitingPolicy && target_fresh) {
        state_.store(State::Running);
        PublishStatus();
        if (state_fresh) {WarnHeadSwing(s);}
      } else if (state_.load() == State::Running && !target_fresh) {
        Degrade("joint_target stale");
      }

      if (state_fresh) {
        SendCommand(s, target, state_.load() == State::Running && target_fresh);
      }

      if (++cmd_window_count_ >= static_cast<uint64_t>(cmd_rate_hz_)) {
        const double dt = duration<double>(now - cmd_window_).count();
        if (dt > 0.0) {cmd_hz_.store(cmd_window_count_ / dt);}
        cmd_window_count_ = 0;
        cmd_window_ = now;
      }

      std::this_thread::sleep_until(next);
      // A long stall must not turn into a burst of catch-up commands.
      if (next < steady_clock::now() - period) {next = steady_clock::now();}
    }
  }

  // `drive` false => damping: zero stiffness, hold nothing, just resist motion.
  // With no R1 client in the SDK (/usr/local/include/unitree/robot has a2, b2,
  // g1, go2, h1 -- no r1) there is no vendor damping/e-stop call to delegate to,
  // so this is the degrade behavior, implemented here.
  void SendCommand(const RobotState & s, const std::vector<float> & target, bool drive)
  {
    // Slots left at mode 0 are the nine G1 positions R1 has no motor for
    // (14, 20, 21, 27, 28, 31-34). Enabling a motor that is not there is not
    // something to try on a first run.
    std::array<MotorCommand, kSlots> cmd{};

    for (std::size_t j = 0; j < kNumJoints; ++j) {
      auto & m = cmd[kJointSlot[j]];
      m.mode = 1;
      m.q = s.motor[kJointSlot[j]].q;   // undriven: hold where it is
      // Damping only, at the joint's own trained kd. A global damping constant
      // is either negligible on a hip or oscillatory on the head.
      m.kd = static_cast<float>(kKd[j] * kd_scale_);
    }

    if (drive) {
      for (std::size_t a = 0; a < kNumActions; ++a) {
        const std::size_t j = kActionToArt[a];
        auto & m = cmd[kJointSlot[j]];
        m.q = target[a];
        m.kp = static_cast<float>(kKp[j] * kp_scale_);
        m.kd = static_cast<float>(kKd[j] * kd_scale_);
      }
      // The head is not actuated by the policy and was fixed during training,
      // so hold it at the default pose rather than leaving it to flop.
      for (std::size_t j = 0; j < kNumJoints; ++j) {
        if (!IsActuated(j)) {
          auto & m = cmd[kJointSlot[j]];
          m.q = kDefaultPos[j] + joint_zero_[j];
          m.kp = static_cast<float>(kKp[j] * kp_scale_);
          m.kd = static_cast<float>(kKd[j] * kd_scale_);
        }
      }
    }

    io_->Send(cmd, s.mode_machine);

    if (++cmd_count_ % static_cast<uint64_t>(cmd_rate_hz_ / 10.0 + 1) == 0) {
      std_msgs::msg::Float32MultiArray dbg;
      dbg.data.reserve(kNumJoints);
      for (std::size_t j = 0; j < kNumJoints; ++j) {
        dbg.data.push_back(cmd[kJointSlot[j]].q);
      }
      dbg_cmd_pub_->publish(dbg);
    }
  }

  // Driving starts by pulling the head to its hold pose at full stiffness. A big
  // step there means head_*_zero does not match this robot.
  void WarnHeadSwing(const RobotState & s)
  {
    for (std::size_t j : {kHeadPitch, kHeadYaw}) {
      const float step = kDefaultPos[j] + joint_zero_[j] - s.motor[kJointSlot[j]].q;
      if (std::fabs(step) > 0.15f) {
        RCLCPP_WARN(get_logger(),
          "%s jumps %+.2f rad as driving starts: check head_pitch_zero / head_yaw_zero "
          "(the reading with the head facing straight ahead)",
          j == kHeadYaw ? "head_yaw" : "head_pitch", step);
      }
    }
  }

  static bool IsActuated(std::size_t art)
  {
    for (std::size_t a = 0; a < kNumActions; ++a) {
      if (static_cast<std::size_t>(kActionToArt[a]) == art) {return true;}
    }
    return false;
  }

  // Rates are reported by the node itself rather than left to `ros2 topic hz`:
  // the data topics are best-effort and foxy's CLI subscribes reliable by
  // default, so the CLI shows an empty topic whether or not anything is wrong.
  void OnStatusTick()
  {
    PublishStatus();
    if (++status_ticks_ % 5 != 0) {return;}
    const auto lag = LoopLag();
    const double step_ms = 1000.0 / control_rate_hz_;
    RCLCPP_INFO(get_logger(),
      "%s | obs %.1f Hz (want %.0f) | cmd %.1f Hz (want %.0f) | output=%s | "
      "kp_scale=%.2f | crc_fail=%zu | setpoint lag p50=%.1f p95=%.1f max=%.1f ms "
      "(%.2f steps; trained 0-%.0f)",
      StateName(state_.load()), obs_hz_.load(), control_rate_hz_,
      cmd_hz_.load(), cmd_rate_hz_,
      io_->output_enabled() ? "ON" : "off", kp_scale_, io_->crc_failures(),
      lag[0], lag[1], lag[2], lag[0] / step_ms, step_ms);
    // Training drew the lag from {0, 1} control steps. Past one step the
    // deployed loop is outside the distribution the policy was trained on, and
    // that is a W07 gap-mitigation input, not a crash -- so say it, once every
    // status cycle, rather than degrading on it.
    if (HaveReflex()) {LogReflexStats();}
    if (lag[1] > step_ms) {
      RCLCPP_WARN(get_logger(),
        "setpoint lag p95=%.1f ms exceeds the trained range (0-%.0f ms = 0-1 "
        "control steps): the policy is running outside its trained lag "
        "distribution", lag[1], step_ms);
    }
  }

  // ------------------------------------------------------------------ reflex
  void DeclareReflex()
  {
    const reflex::Config d;
    reflex::Config c;
    c.enable = declare_parameter<bool>("reflex_enable", false);
    c.person_trigger_m = declare_parameter<double>("reflex_person_trigger_m", d.person_trigger_m);
    c.obstacle_trigger_m =
      declare_parameter<double>("reflex_obstacle_trigger_m", d.obstacle_trigger_m);
    c.clear_margin_m = declare_parameter<double>("reflex_clear_margin_m", d.clear_margin_m);
    c.confirm_frames = static_cast<int>(
      declare_parameter<int64_t>("reflex_confirm_frames", d.confirm_frames));
    c.retreat_vx = declare_parameter<double>("reflex_retreat_vx", d.retreat_vx);
    c.retreat_s = declare_parameter<double>("reflex_retreat_s", d.retreat_s);
    c.max_retreats = static_cast<int>(
      declare_parameter<int64_t>("reflex_max_retreats", d.max_retreats));
    c.window_s = declare_parameter<double>("reflex_window_s", d.window_s);
    c.stale_s = declare_parameter<double>("reflex_stale_s", d.stale_s);
    c.center_valid_min = declare_parameter<double>("reflex_center_valid_min", d.center_valid_min);
    c.release_margin_m = declare_parameter<double>("reflex_release_margin_m", d.release_margin_m);
    c.person_floor_m = declare_parameter<double>("reflex_person_floor_m", d.person_floor_m);
    c.obstacle_floor_m = declare_parameter<double>("reflex_obstacle_floor_m", d.obstacle_floor_m);
    c.stop_margin_m = declare_parameter<double>("reflex_stop_margin_m", d.stop_margin_m);
    c.latency_s = declare_parameter<double>("reflex_latency_s", d.latency_s);
    c.k_lat = declare_parameter<double>("reflex_k_lat", d.k_lat);
    c.k_brake = declare_parameter<double>("reflex_k_brake", d.k_brake);
    c.speed_window_s = declare_parameter<double>("reflex_speed_window_s", d.speed_window_s);
    reflex_topic_ = declare_parameter<std::string>("reflex_proximity_topic", "/r1_perception/proximity");
    const auto log_path = declare_parameter<std::string>("reflex_log", "");
    const reflex::DanceConfig dd;
    reflex::DanceConfig dc;
    const bool dance = declare_parameter<bool>("dance_enable", false);
    dc.person_trigger_m = declare_parameter<double>("dance_person_trigger_m", dd.person_trigger_m);
    dc.obstacle_trigger_m = declare_parameter<double>("dance_obstacle_trigger_m", dd.obstacle_trigger_m);
    dc.clear_m = declare_parameter<double>("dance_clear_m", dd.clear_m);
    dc.still_s = declare_parameter<double>("dance_still_s", dd.still_s);
    dc.heartbeat_s = declare_parameter<double>("dance_heartbeat_s", dd.heartbeat_s);
    dc.max_s = declare_parameter<double>("dance_max_s", dd.max_s);
    dance_target_stale_s_ = declare_parameter<double>("dance_target_stale_s", 0.06);
    if (!c.enable) {
      if (dance) {
        throw std::invalid_argument(
          "dance_enable needs reflex_enable: a dancing robot must step back from people");
      }
      RCLCPP_INFO(get_logger(), "reflex_enable=false -- no proximity reflex");
      return;
    }
    // The retreat is a command like any other: it has to be inside what the policy
    // was trained on. Under v1's envelope (forward only) there is no retreat.
    if (!(c.retreat_vx < 0.0) || c.retreat_vx < cmd_vel_x_range_[0]) {
      throw std::invalid_argument(
        "reflex_retreat_vx must be negative and inside cmd_vel_x_range (the policy's trained "
        "range); a policy that was never trained to walk backward cannot retreat");
    }
    if (dance) {
      conductor_.emplace(c, dc);
    } else {
      reflex_.emplace(c);
    }
    if (!log_path.empty()) {
      reflex_log_.open(log_path, std::ios::out | std::ios::trunc);
      if (!reflex_log_) {throw std::runtime_error("cannot write reflex_log " + log_path);}
      reflex_log_ << "t,state,reason,blocked,refused,cmd_vx,cmd_vy,cmd_wz,out_vx,out_vy,out_wz,"
        "frame_seq,frame_t_cap,frame_age_s,person_m,person_near_box,obstacle_m,depth_ok,"
        "center_valid,new_frame,reset" << (dance ? ",policy,dance_tick,dance_state,dance_reason,"
        "dance_msg,dance_ok" : "") << "\n";
    }
    RCLCPP_WARN(get_logger(),
      "reflex ON: retreat %.2f m/s for %.2f s when a person < %.2f m or an obstacle < %.2f m on "
      "%d frames; stop distance person %.2f/%.2f/%.2f m, obstacle %.2f/%.2f/%.2f m at "
      "0/0.3/1.0 m/s; frames from %s%s%s",
      c.retreat_vx, c.retreat_s, c.person_trigger_m, c.obstacle_trigger_m, c.confirm_frames,
      Rx().StopPerson(0.0), Rx().StopPerson(0.3), Rx().StopPerson(1.0),
      Rx().StopObstacle(0.0), Rx().StopObstacle(0.3), Rx().StopObstacle(1.0),
      reflex_topic_.c_str(), log_path.empty() ? "" : "; log ", log_path.c_str());
    if (dance) {
      RCLCPP_WARN(get_logger(),
        "dance ON: two policies; a dance starts on ~/dance with nobody within %.2f m and the "
        "robot standing %.1f s; dancing, a person or obstacle < %.2f / %.2f m steps back at "
        "once; heartbeat %.2f s, at most %.0f s; the dance policy's output must be younger "
        "than %.0f ms", dc.clear_m, dc.still_s, dc.person_trigger_m, dc.obstacle_trigger_m,
        dc.heartbeat_s, dc.max_s, dance_target_stale_s_ * 1e3);
    }
  }

  bool HaveReflex() const {return reflex_.has_value() || conductor_.has_value();}
  // The reflex state machine: the conductor's own when dancing is on.
  const reflex::Fsm & Rx() const {return conductor_ ? conductor_->reflex() : *reflex_;}

  void OnDanceTarget(const std_msgs::msg::Float32MultiArray & m)
  {
    if (m.data.size() != kNumActions) {return;}
    bool finite = true;
    for (const float v : m.data) {finite = finite && std::isfinite(v);}
    std::lock_guard<std::mutex> lock(target_mutex_);
    if (!finite) {
      // Not a reason to degrade the bridge: v2 is running and takes over (the
      // conductor sees the dance's output as not ok).
      dance_target_bad_ = true;
      return;
    }
    std::copy(m.data.begin(), m.data.end(), dance_target_.begin());
    last_dance_target_time_ = steady_clock::now();
    have_dance_target_ = true;
    dance_target_bad_ = false;
  }

  void OnProximity(const std_msgs::msg::Float64MultiArray & m)
  {
    const auto f = reflex::ParseProximity(m.data);
    if (!f) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000,
        "%s: %zu values, version %.0f -- expected %zu, version %.0f; frame ignored "
        "(perception and bridge from different V3 builds?)", reflex_topic_.c_str(),
        m.data.size(), m.data.empty() ? -1.0 : m.data[0], reflex::kProxLen, reflex::kProxVersion);
      return;
    }
    const double t = duration<double>(steady_clock::now().time_since_epoch()).count();
    std::lock_guard<std::mutex> lock(reflex_mutex_);
    reflex_frame_ = *f;
    ++reflex_frames_;
    reflex_ages_.push_back(t - f->t_cap);
  }

  // The reflex's step: `cmd` is what the observation would get; it is replaced by
  // what the reflex allows. steady_clock is CLOCK_MONOTONIC, the clock perception
  // stamps frames with (Python time.monotonic()).
  void ApplyReflex(steady_clock::time_point now, std::array<float, 3> & cmd)
  {
    const double t = duration<double>(now.time_since_epoch()).count();
    std::optional<reflex::Frame> frame;
    bool reset = false;
    reflex::DanceMsg dmsg = reflex::DanceMsg::None;
    {
      std::lock_guard<std::mutex> lock(reflex_mutex_);
      frame.swap(reflex_frame_);
      reset = reflex_reset_pending_;
      reflex_reset_pending_ = false;
      dmsg = dance_pending_;
      dance_pending_ = reflex::DanceMsg::None;
    }
    const bool running = state_.load() == State::Running;
    const auto before = Rx().state();
    reflex::Output o;
    bool dance_ok = false;
    if (conductor_) {
      {
        std::lock_guard<std::mutex> lock(target_mutex_);
        dance_ok = have_dance_target_ && !dance_target_bad_ &&
          duration<double>(now - last_dance_target_time_).count() < dance_target_stale_s_;
      }
      const auto before_policy = conductor_->policy();
      const auto co = conductor_->Step(t, cmd[0], cmd[1], cmd[2], frame ? &*frame : nullptr, reset,
                                       running, dmsg, dance_ok);
      o = co.reflex;
      policy_.store(co.policy);
      dance_tick_ = co.dance_tick;
      if (co.policy != before_policy || co.dance_state != last_dance_.dance_state ||
        co.dance_reason != last_dance_.dance_reason)
      {
        RCLCPP_WARN(get_logger(), "DANCE: policy %s -> %s, %s (%s)",
                    reflex::PolicyName(before_policy), reflex::PolicyName(co.policy),
                    reflex::DanceStateName(co.dance_state), reflex::DanceReasonText(co.dance_reason));
      }
      dance_changed_ = co.policy != last_dance_.policy || co.dance_state != last_dance_.dance_state ||
        co.dance_reason != last_dance_.dance_reason;
      last_dance_ = co;
    } else {
      o = reflex_->Step(t, cmd[0], cmd[1], cmd[2], frame ? &*frame : nullptr, reset, running);
    }
    const std::array<float, 3> in = cmd;
    cmd = {static_cast<float>(o.vx), static_cast<float>(o.vy), static_cast<float>(o.wz)};

    const auto & f = Rx().frame();
    const double age = f ? t - f->t_cap : -1.0;
    if (o.state == reflex::State::Retreat && before != reflex::State::Retreat) {
      // The latency the stop distance assumes: from the frame's capture to the
      // retreat command in the observation (this one).
      RCLCPP_WARN(get_logger(),
        "REFLEX: %s -- person %s, obstacle %s; frame %ld captured %.0f ms ago; retreat %zu in %.0f s",
        reflex::ReasonText(o.reason), Fmt(f ? f->person_m : std::nullopt, f && f->person_near_box).c_str(),
        Fmt(f && f->depth_ok ? f->obstacle_m : std::nullopt, false).c_str(),
        f ? f->seq : -1L, age * 1e3, Rx().retreats().size(), Rx().config().window_s);
      ++reflex_retreats_total_;
    } else if (o.state != before || o.reset_refused) {
      RCLCPP_INFO(get_logger(), "REFLEX: %s -> %s (%s)", reflex::StateName(before),
                  reflex::StateName(o.state), reflex::ReasonText(o.reason));
    }

    const bool changed = o.state != last_out_.state || o.reason != last_out_.reason ||
      o.blocked != last_out_.blocked || o.reset_refused || dance_changed_;
    if (changed || t - reflex_pub_t_ >= 0.5) {
      PublishReflex(t, o, in, age);
      reflex_pub_t_ = t;
    }
    last_out_ = o;

    if (reflex_log_.is_open()) {
      char buf[512];
      std::snprintf(buf, sizeof(buf),
        "%.6f,%s,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%ld,%.6f,%.4f,%s,%d,%s,%d,%.3f,%d,%d",
        t, reflex::StateName(o.state), o.reason, o.blocked, o.reset_refused, in[0], in[1], in[2],
        o.vx, o.vy, o.wz, f ? f->seq : -1L, f ? f->t_cap : 0.0, age,
        Num(f ? f->person_m : std::nullopt).c_str(), f ? f->person_near_box : 0,
        Num(f ? f->obstacle_m : std::nullopt).c_str(), f ? f->depth_ok : 0,
        f ? f->center_valid : 0.0, frame ? 1 : 0, reset ? 1 : 0);
      reflex_log_ << buf;
      if (conductor_) {
        std::snprintf(buf, sizeof(buf), ",%s,%ld,%s,%d,%d,%d",
          reflex::PolicyName(last_dance_.policy), last_dance_.dance_tick,
          reflex::DanceStateName(last_dance_.dance_state), last_dance_.dance_reason,
          dmsg == reflex::DanceMsg::Dance ? 1 : (dmsg == reflex::DanceMsg::End ? 0 : -1),
          dance_ok ? 1 : 0);
        reflex_log_ << buf;
      }
      reflex_log_ << "\n";
    }
  }

  // ~/reflex, std_msgs/String JSON: what mission_ctl acts on and reports.
  void PublishReflex(double t, const reflex::Output & o, const std::array<float, 3> & in, double age)
  {
    const auto & f = Rx().frame();
    const auto & d = last_dance_;
    char buf[1000];
    std::snprintf(buf, sizeof(buf),
      "{\"v\":1,\"t\":%.6f,\"state\":\"%s\",\"reason\":%d,\"text\":\"%s\",\"blocked\":%s,"
      "\"reset_refused\":%s,\"cmd_in\":[%.4f,%.4f,%.4f],\"cmd_out\":[%.4f,%.4f,%.4f],"
      "\"frame_seq\":%ld,\"frame_age_s\":%.4f,\"person_m\":%s,\"person_near_box\":%s,"
      "\"obstacle_m\":%s,\"depth_ok\":%s,\"retreats_in_window\":%zu,"
      "\"policy\":\"%s\",\"dance_state\":\"%s\",\"dance_reason\":%d,\"dance_tick\":%ld,"
      "\"dance_text\":\"%s\"}",
      t, reflex::StateName(o.state), o.reason, reflex::ReasonText(o.reason),
      o.blocked ? "true" : "false", o.reset_refused ? "true" : "false", in[0], in[1], in[2],
      o.vx, o.vy, o.wz, f ? f->seq : -1L, age, Json(f ? f->person_m : std::nullopt).c_str(),
      f && f->person_near_box ? "true" : "false", Json(f ? f->obstacle_m : std::nullopt).c_str(),
      f && f->depth_ok ? "true" : "false", Rx().retreats().size(),
      reflex::PolicyName(d.policy), reflex::DanceStateName(d.dance_state), d.dance_reason,
      d.dance_tick, reflex::DanceReasonText(d.dance_reason));
    std_msgs::msg::String m;
    m.data = buf;
    reflex_pub_->publish(m);
  }

  // Every 5 s with the status line: how fresh the frames are when they arrive.
  void LogReflexStats()
  {
    std::vector<double> ages;
    uint64_t frames;
    {
      std::lock_guard<std::mutex> lock(reflex_mutex_);
      ages.swap(reflex_ages_);
      frames = reflex_frames_;
      reflex_frames_ = 0;
    }
    std::sort(ages.begin(), ages.end());
    auto at = [&ages](double q) {
        return ages.empty() ? 0.0 :
               ages[std::min(ages.size() - 1, static_cast<std::size_t>(q * ages.size()))] * 1e3;
      };
    RCLCPP_INFO(get_logger(),
      "REFLEX %s | frames %.1f Hz | age on arrival ms p50=%.0f p95=%.0f p99=%.0f max=%.0f | "
      "retreats %lu",
      reflex::StateName(Rx().state()), frames / 5.0, at(0.50), at(0.95), at(0.99),
      ages.empty() ? 0.0 : ages.back() * 1e3, static_cast<unsigned long>(reflex_retreats_total_));
    if (frames == 0) {
      RCLCPP_WARN(get_logger(), "REFLEX: no frames on %s -- forward walking is blocked "
                  "(is r1_perception running?)", reflex_topic_.c_str());
    }
  }

  static std::string Fmt(const std::optional<double> & v, bool near_box)
  {
    if (near_box) {return "near (box)";}
    if (!v) {return "none";}
    char b[32];
    std::snprintf(b, sizeof(b), "%.2f m", *v);
    return b;
  }

  static std::string Num(const std::optional<double> & v)
  {
    if (!v) {return "nan";}
    char b[32];
    std::snprintf(b, sizeof(b), "%.4f", *v);
    return b;
  }

  static std::string Json(const std::optional<double> & v) {return v ? Num(v) : "null";}

  void PublishStatus()
  {
    std_msgs::msg::String m;
    m.data = std::string(StateName(state_.load())) +
      " output=" + (io_->output_enabled() ? "ON" : "off") +
      " crc_fail=" + std::to_string(io_->crc_failures()) +
      (state_.load() == State::Degraded ? " reason=" + Reason() : "");
    status_pub_->publish(m);
    if (auto_recover_ && state_.load() == State::Degraded) {
      RCLCPP_WARN_ONCE(get_logger(),
                       "auto_recover=true: degraded state will NOT self-clear; "
                       "publish true to ~/resume. The parameter exists only to "
                       "make its absence explicit.");
    }
  }

  // parameters
  double control_rate_hz_ = 50.0, cmd_rate_hz_ = 500.0;
  double kp_scale_ = 1.0, kd_scale_ = 1.0;
  double state_timeout_ms_ = 100.0, target_timeout_ms_ = 60.0, min_rate_hz_ = 45.0;
  double cmd_vel_timeout_ms_ = 500.0;
  bool auto_recover_ = false;

  std::unique_ptr<RobotIo> io_;

  std::mutex state_mutex_;
  RobotState latest_;
  bool have_state_ = false;
  steady_clock::time_point last_state_time_;

  std::mutex target_mutex_;
  std::vector<float> target_, last_action_;
  std::array<float, 3> cmd_vel_{};
  bool have_target_ = false, have_cmdvel_ = false;
  steady_clock::time_point last_target_time_, last_cmdvel_time_;

  std::vector<float> frame_;
  std::array<float, kNumJoints> joint_zero_{};   // head_pitch_zero / head_yaw_zero; 0 elsewhere
  std_msgs::msg::Float32MultiArray obs_msg_;
  // Read and written from both the ROS executor and the command thread.
  std::atomic<State> state_{State::WaitingState};
  std::mutex reason_mutex_;
  std::string degrade_reason_;
  std::atomic<bool> running_{true};
  std::thread cmd_thread_;
  std::vector<double> cmd_vel_x_range_, cmd_vel_y_range_, cmd_vel_yaw_range_;
  uint64_t cmd_count_ = 0, obs_count_ = 0, cmd_window_count_ = 0, status_ticks_ = 0;
  int slow_windows_ = 0;
  bool rate_primed_ = false;
  static constexpr std::size_t kLagSamples = 512;
  std::atomic<int64_t> obs_pub_ns_{0};
  std::mutex lag_mutex_;
  std::vector<double> lag_samples_;
  std::size_t lag_next_ = 0;
  steady_clock::time_point rate_window_ = steady_clock::now();
  steady_clock::time_point cmd_window_ = steady_clock::now();
  std::atomic<double> obs_hz_{0.0}, cmd_hz_{0.0};

  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr obs_pub_,
    dbg_jpos_pub_, dbg_jvel_pub_, dbg_imu_pub_, dbg_cmd_pub_, dbg_tau_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr reset_pub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr
    target_sub_, action_sub_, cmdvel_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr resume_sub_;
  rclcpp::TimerBase::SharedPtr obs_timer_, status_timer_;

  // reflex (V3); reflex_ is empty when reflex_enable is false
  std::optional<reflex::Fsm> reflex_;
  std::string reflex_topic_;
  std::mutex reflex_mutex_;
  std::optional<reflex::Frame> reflex_frame_;   // newest frame not yet stepped
  bool reflex_reset_pending_ = false;
  uint64_t reflex_frames_ = 0, reflex_retreats_total_ = 0;
  std::vector<double> reflex_ages_;
  reflex::Output last_out_;
  double reflex_pub_t_ = 0.0;
  std::ofstream reflex_log_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr prox_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reflex_reset_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr reflex_pub_;

  // V3 dance; conductor_ is empty when dance_enable is false (and then reflex_ is used)
  std::optional<reflex::Conductor> conductor_;
  std::atomic<reflex::Policy> policy_{reflex::Policy::Walk};
  reflex::Policy applied_policy_ = reflex::Policy::Walk;   // obs thread only
  long dance_tick_ = -1;                                   // obs thread only
  reflex::DanceMsg dance_pending_ = reflex::DanceMsg::None;   // under reflex_mutex_
  reflex::ConductorOutput last_dance_;
  bool dance_changed_ = false;
  double dance_target_stale_s_ = 0.06;
  std::vector<float> dance_target_, dance_last_action_;     // under target_mutex_
  bool have_dance_target_ = false, dance_target_bad_ = false;
  steady_clock::time_point last_dance_target_time_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr dance_sub_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr dance_target_sub_,
    dance_action_sub_;
};

}  // namespace r1_hw_bridge

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<r1_hw_bridge::BridgeNode>());
  rclcpp::shutdown();
  return 0;
}
