// ROS 2 node running the R1 locomotion policy through TensorRT.
//
// Division of labour, which is the point of this node's shape:
//
//   upstream (W06: robot / sim bridge)  ->  ~/obs        one 85-float frame per
//                                                        control cycle, current
//                                                        sensor values only
//   THIS NODE                           ->  history stacking, inference
//   downstream (W06: joint command)     <-  ~/action     24 raw policy outputs
//                                       <-  ~/joint_target  (optional) the same
//                                                        actions mapped through
//                                                        default + scale
//
// Keeping the 5-frame history inside this node rather than in the publisher is
// deliberate: the term-major layout the policy expects is an artifact of how it
// was trained, and every consumer that re-derives it is another chance to get it
// silently wrong. See obs_assembler.hpp.
//
// V3 (this fork, v3/ros2_ws): a second engine, the dance policy, when
// `dance_engine_path` is set. Then ~/obs carries 6 more floats (the IMU quaternion, the
// dance tick, the active policy: dance_obs.hpp), and on every frame BOTH engines run,
// the walking policy first: v2 as before on the first 85 floats, the dance policy on the
// observation dance_obs.hpp builds with the dance's reference. The dance's outputs go
// out on ~/dance_action and ~/dance_joint_target; the bridge chooses which reaches the
// motors (its conductor). Without `dance_engine_path` nothing here differs from the
// original node.

#include "r1_policy_runner/dance_obs.hpp"
#include "r1_policy_runner/obs_assembler.hpp"
#include "r1_policy_runner/trt_policy.hpp"

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace r1_policy_runner
{

using std::chrono::duration;
using std::chrono::steady_clock;

class PolicyNode : public rclcpp::Node
{
public:
  PolicyNode()
  : rclcpp::Node("r1_policy_node")
  {
    const auto engine_path = declare_parameter<std::string>("engine_path", "");
    const auto mode = declare_parameter<std::string>("mode", "subscribe");
    rate_hz_ = declare_parameter<double>("control_rate_hz", 50.0);
    const int64_t history = declare_parameter<int64_t>("history_length", 5);
    const auto term_names = declare_parameter<std::vector<std::string>>(
      "obs_term_names", std::vector<std::string>{});
    const auto term_widths = declare_parameter<std::vector<int64_t>>(
      "obs_term_widths", std::vector<int64_t>{});
    default_joint_pos_ = declare_parameter<std::vector<double>>(
      "default_joint_pos", std::vector<double>{});
    action_scale_ = declare_parameter<std::vector<double>>(
      "action_scale", std::vector<double>{});
    stats_period_s_ = declare_parameter<double>("stats_period_s", 5.0);
    const auto warmup_hold = declare_parameter<bool>("hold_until_warm", true);
    hold_until_warm_ = warmup_hold;
    const auto dance_engine_path = declare_parameter<std::string>("dance_engine_path", "");
    const auto dance_reference_path = declare_parameter<std::string>("dance_reference_path", "");

    if (engine_path.empty()) {
      throw std::runtime_error("parameter 'engine_path' is required");
    }
    if (term_names.size() != term_widths.size() || term_names.empty()) {
      throw std::runtime_error(
              "'obs_term_names' and 'obs_term_widths' must be non-empty and the same length "
              "-- generate them with tools/dump_interface.py");
    }

    std::vector<ObsTermSpec> terms;
    terms.reserve(term_names.size());
    for (std::size_t i = 0; i < term_names.size(); ++i) {
      terms.push_back({term_names[i], static_cast<std::size_t>(term_widths[i])});
    }
    assembler_ = std::make_unique<ObsAssembler>(std::move(terms), static_cast<std::size_t>(history));

    policy_ = std::make_unique<TrtPolicy>(engine_path);

    // The engine is the authority on dimensions; the parameters describe how to
    // fill them. Disagreement means the params file and the checkpoint came
    // from different training runs, which must not be recoverable at runtime.
    if (static_cast<std::size_t>(policy_->input_dim()) != assembler_->obs_dim()) {
      throw std::runtime_error(
              "observation layout does not match the engine: params describe " +
              std::to_string(assembler_->frame_dim()) + " x " + std::to_string(history) +
              " = " + std::to_string(assembler_->obs_dim()) + ", engine expects " +
              std::to_string(policy_->input_dim()));
    }

    action_dim_ = static_cast<std::size_t>(policy_->output_dim());
    emit_targets_ = default_joint_pos_.size() == action_dim_ &&
      action_scale_.size() == action_dim_;
    if (!default_joint_pos_.empty() && !emit_targets_) {
      RCLCPP_WARN(
        get_logger(),
        "default_joint_pos/action_scale present but not %zu long; not publishing joint_target",
        action_dim_);
    }

    if (!dance_engine_path.empty()) {
      if (dance_reference_path.empty()) {
        throw std::runtime_error("dance_engine_path is set but dance_reference_path is not");
      }
      if (assembler_->frame_dim() != kV2Frame) {
        throw std::runtime_error("the dance needs the walking policy's 85-float frame");
      }
      dance_policy_ = std::make_unique<TrtPolicy>(dance_engine_path);
      dance_ref_ = std::make_unique<DanceReference>(dance_reference_path);
      dance_obs_builder_ = std::make_unique<DanceObsBuilder>(*dance_ref_);
      if (static_cast<std::size_t>(dance_policy_->input_dim()) != kDanceObs ||
        static_cast<std::size_t>(dance_policy_->output_dim()) != action_dim_)
      {
        throw std::runtime_error(
                "dance engine is " + std::to_string(dance_policy_->input_dim()) + " -> " +
                std::to_string(dance_policy_->output_dim()) + ", expected " +
                std::to_string(kDanceObs) + " -> " + std::to_string(action_dim_));
      }
      dance_obs_.assign(kDanceObs, 0.0f);
      dance_action_.assign(action_dim_, 0.0f);
      dance_action_msg_.data.resize(action_dim_);
      dance_target_msg_.data.resize(action_dim_);
      RCLCPP_INFO(get_logger(), "dance engine '%s' loaded (plan %s); reference %s: %zu frames",
        dance_engine_path.c_str(), r1_policy_runner::PlanFingerprint(dance_engine_path).c_str(),
        dance_reference_path.c_str(), dance_ref_->frames());
    }
    in_dim_ = dance_policy_ ? kDanceFrame : assembler_->frame_dim();

    frame_.assign(in_dim_, 0.0f);
    obs_.assign(assembler_->obs_dim(), 0.0f);
    action_.assign(action_dim_, 0.0f);

    // Preallocate the outgoing messages: a control loop should not be asking
    // the allocator for memory every cycle.
    action_msg_.data.resize(action_dim_);
    target_msg_.data.resize(action_dim_);

    const auto qos = rclcpp::SensorDataQoS().keep_last(1);
    action_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/action", qos);
    latency_pub_ = create_publisher<std_msgs::msg::Float32>("~/latency_us", qos);
    if (emit_targets_) {
      target_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/joint_target", qos);
    }
    if (dance_policy_) {
      dance_action_pub_ = create_publisher<std_msgs::msg::Float32MultiArray>("~/dance_action", qos);
      if (emit_targets_) {
        dance_target_pub_ =
          create_publisher<std_msgs::msg::Float32MultiArray>("~/dance_joint_target", qos);
      }
    }
    status_pub_ = create_publisher<std_msgs::msg::String>(
      "~/status", rclcpp::QoS(rclcpp::KeepLast(10)).reliable().transient_local());

    // The bridge asks for this after an outage. Recovery cannot be automatic:
    // the 5-frame history now straddles a gap, which the policy never saw in
    // training, so the buffer is cleared and refilled before output resumes.
    reset_sub_ = create_subscription<std_msgs::msg::Empty>(
      "~/policy_reset", rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
      [this](std_msgs::msg::Empty::ConstSharedPtr) {OnReset();});

    RCLCPP_INFO(
      get_logger(),
      "engine '%s' loaded: %s[%ld] -> %s[%ld]; obs = %zu terms x %ld frames = %zu",
      engine_path.c_str(),
      policy_->info().input_name.c_str(), policy_->input_dim(),
      policy_->info().output_name.c_str(), policy_->output_dim(),
      assembler_->terms().size(), history, assembler_->obs_dim());

    // Engine builds are not reproducible (tactic selection is remeasured, and
    // TF32 can silently win a rebuild), so a parity PASS certifies one file.
    // Log its fingerprint: this is how you prove the engine that passed the
    // gate is the engine now driving the motors.
    RCLCPP_INFO(
      get_logger(), "engine plan %s",
      r1_policy_runner::PlanFingerprint(engine_path).c_str());

    if (mode == "subscribe") {
      obs_sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
        "~/obs", qos,
        [this](std_msgs::msg::Float32MultiArray::ConstSharedPtr msg) {OnObs(*msg);});
      RCLCPP_INFO(get_logger(), "mode=subscribe: waiting for ~/obs (%zu floats per frame)",
        in_dim_);
    } else if (mode == "selftest") {
      // Free-running rehearsal with no robot attached: the node drives itself
      // at the control rate on a synthetic frame. This is what makes W06's
      // hanging dry-run testable before the robot is on the bench.
      timer_ = create_wall_timer(
        std::chrono::duration<double>(1.0 / rate_hz_),
        [this]() {OnSelfTestTick();});
      RCLCPP_INFO(get_logger(), "mode=selftest: self-driving at %.1f Hz", rate_hz_);
    } else {
      throw std::runtime_error("parameter 'mode' must be 'subscribe' or 'selftest'");
    }

    last_stats_ = steady_clock::now();
  }

private:
  void OnReset()
  {
    assembler_->Reset();
    if (dance_obs_builder_) {dance_obs_builder_->Reset();}
    // Forced on regardless of the parameter: resuming mid-gap is exactly the
    // case the parameter must not be allowed to disable.
    hold_until_warm_ = true;
    nan_outputs_ = 0;
    RCLCPP_WARN(get_logger(), "reset: history cleared, re-warming %zu frames",
      assembler_->history());
    PublishStatus("REWARMING");
  }

  void PublishStatus(const std::string & what)
  {
    if (!status_pub_) {return;}
    std_msgs::msg::String m;
    m.data = what + " infer_failures=" + std::to_string(infer_failures_) +
      " nan_outputs=" + std::to_string(nan_outputs_);
    status_pub_->publish(m);
  }

  void OnObs(const std_msgs::msg::Float32MultiArray & msg)
  {
    if (msg.data.size() != in_dim_) {
      // Throttled: a mis-sized publisher would otherwise flood the log at 50 Hz.
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "dropping ~/obs frame: got %zu floats, expected %zu%s",
        msg.data.size(), in_dim_,
        dance_policy_ ? " (the dance needs the V3 bridge with dance_enable true)" : "");
      return;
    }
    std::copy(msg.data.begin(), msg.data.end(), frame_.begin());
    Step();
    if (dance_policy_) {DanceStep();}
  }

  void OnSelfTestTick()
  {
    // A slow sinusoid across the frame keeps successive observations distinct,
    // so a history-stacking bug shows up as a constant action rather than
    // hiding behind identical inputs.
    const double t = duration<double>(steady_clock::now() - start_).count();
    for (std::size_t i = 0; i < frame_.size(); ++i) {
      frame_[i] = static_cast<float>(0.05 * std::sin(2.0 * M_PI * 0.5 * t + 0.1 * i));
    }
    Step();
  }

  void Step()
  {
    const auto t0 = steady_clock::now();

    assembler_->Push(frame_.data());
    if (hold_until_warm_ && !assembler_->warm()) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "filling observation history (%zu frames needed)", assembler_->history());
      return;
    }
    assembler_->Fill(obs_.data());

    if (!policy_->Infer(obs_.data(), action_.data())) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "TensorRT inference failed");
      ++infer_failures_;
      PublishStatus("INFER_FAILED");
      return;
    }

    // A NaN reaches the motors as a command like any other number. Withholding
    // the message is what puts the bridge into damping: its target-stale
    // watchdog fires after 3 cycles. Publishing a "safe" substitute here would
    // hide the fault from the layer that owns the safety response.
    for (const float a : action_) {
      if (!std::isfinite(a)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
          "policy output contains NaN/inf -- withholding action");
        ++nan_outputs_;
        PublishStatus("NAN_OUTPUT");
        return;
      }
    }

    const double us = duration<double, std::micro>(steady_clock::now() - t0).count();
    latencies_.push_back(us);

    action_msg_.data.assign(action_.begin(), action_.end());
    action_pub_->publish(action_msg_);

    if (emit_targets_) {
      for (std::size_t i = 0; i < action_dim_; ++i) {
        target_msg_.data[i] = static_cast<float>(
          default_joint_pos_[i] + action_scale_[i] * action_[i]);
      }
      target_pub_->publish(target_msg_);
    }

    latency_msg_.data = static_cast<float>(us);
    latency_pub_->publish(latency_msg_);

    ++cycles_;
    MaybeReportStats();
  }

  // The dance policy on this frame: built, inferred, published. A NaN is withheld like
  // the walking policy's; the bridge sees the dance's output go stale and hands the
  // robot to the walking policy (conductor.hpp: kStopFault).
  void DanceStep()
  {
    const auto t0 = steady_clock::now();
    dance_obs_builder_->Build(frame_.data(), dance_obs_.data());
    if (!dance_policy_->Infer(dance_obs_.data(), dance_action_.data())) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "dance: TensorRT inference failed");
      ++infer_failures_;
      return;
    }
    for (const float a : dance_action_) {
      if (!std::isfinite(a)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000,
          "dance policy output contains NaN/inf -- withholding it");
        ++nan_outputs_;
        return;
      }
    }
    dance_latencies_.push_back(duration<double, std::micro>(steady_clock::now() - t0).count());
    dance_action_msg_.data.assign(dance_action_.begin(), dance_action_.end());
    dance_action_pub_->publish(dance_action_msg_);
    if (dance_target_pub_) {
      for (std::size_t i = 0; i < action_dim_; ++i) {
        dance_target_msg_.data[i] = static_cast<float>(
          default_joint_pos_[i] + action_scale_[i] * dance_action_[i]);
      }
      dance_target_pub_->publish(dance_target_msg_);
    }
  }

  void MaybeReportStats()
  {
    const auto now = steady_clock::now();
    const double elapsed = duration<double>(now - last_stats_).count();
    if (elapsed < stats_period_s_ || latencies_.empty()) {return;}

    std::vector<double> sorted = latencies_;
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&sorted](double p) {
        return sorted[static_cast<std::size_t>(p * (sorted.size() - 1))];
      };

    RCLCPP_INFO(
      get_logger(),
      "%zu cycles in %.1fs (%.1f Hz) | infer us p50=%.0f p95=%.0f p99=%.0f max=%.0f | "
      "budget %.0f us | failures=%zu",
      latencies_.size(), elapsed, latencies_.size() / elapsed,
      pct(0.50), pct(0.95), pct(0.99), sorted.back(),
      1e6 / rate_hz_, infer_failures_);

    if (dance_policy_ && !dance_latencies_.empty()) {
      std::vector<double> d = dance_latencies_;
      std::sort(d.begin(), d.end());
      const auto dp = [&d](double p) {return d[static_cast<std::size_t>(p * (d.size() - 1))];};
      RCLCPP_INFO(get_logger(), "dance: %zu cycles | build + infer us p50=%.0f p95=%.0f p99=%.0f max=%.0f",
        d.size(), dp(0.50), dp(0.95), dp(0.99), d.back());
      dance_latencies_.clear();
    }

    PublishStatus(latencies_.size() / elapsed >= 0.9 * rate_hz_ ? "RUNNING" : "SLOW");

    latencies_.clear();
    last_stats_ = now;
  }

  std::unique_ptr<TrtPolicy> policy_;
  // V3 dance (all empty when dance_engine_path is not set)
  std::unique_ptr<TrtPolicy> dance_policy_;
  std::unique_ptr<DanceReference> dance_ref_;
  std::unique_ptr<DanceObsBuilder> dance_obs_builder_;
  std::vector<float> dance_obs_, dance_action_;
  std::vector<double> dance_latencies_;
  std_msgs::msg::Float32MultiArray dance_action_msg_, dance_target_msg_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr dance_action_pub_, dance_target_pub_;
  std::size_t in_dim_ = 0;
  std::unique_ptr<ObsAssembler> assembler_;

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr obs_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr action_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr target_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr latency_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::vector<float> frame_, obs_, action_;
  std::vector<double> default_joint_pos_, action_scale_;
  std_msgs::msg::Float32MultiArray action_msg_, target_msg_;
  std_msgs::msg::Float32 latency_msg_;

  std::size_t action_dim_ = 0;
  bool emit_targets_ = false;
  bool hold_until_warm_ = true;
  double rate_hz_ = 50.0;
  double stats_period_s_ = 5.0;

  std::vector<double> latencies_;
  std::size_t cycles_ = 0;
  std::size_t infer_failures_ = 0;
  std::size_t nan_outputs_ = 0;
  steady_clock::time_point last_stats_;
  const steady_clock::time_point start_ = steady_clock::now();
};

}  // namespace r1_policy_runner

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<r1_policy_runner::PolicyNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("r1_policy_node"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
