// V3 conductor: which policy drives the robot (v2 walking, or the dance), the dance's
// clock, and the reflex around it. NO ROS types: tested with a C++17 compiler alone
// (test/test_reflex_fsm.cpp replays v3/reflex/golden_dance.csv).
//
// A line-for-line port of v3/reflex/conductor.py, which is the reference and explains
// the rules; change the Python first, regenerate golden_dance.csv
// (v3/reflex/dance_cases.py --write), then this. In short, once per control step:
//   * the reflex (reflex_fsm.hpp) runs as before -- given (0, 0, 0) while dancing, so
//     the walking policy, which runs in the background, stands; and with the dance's
//     triggers (0.9 m), because the arms reach;
//   * dancing, a retreat (or the reflex off, or the dance policy's output stale or not
//     finite) hands the robot to the walking policy on that same step;
//   * an end request, a lost heartbeat, the length cap or stale camera frames end the
//     dance at the next bar boundary, where it is back in the standing pose;
//   * a dance starts on a dance request when the reflex is ARMED, a fresh frame shows
//     nothing within dance_clear_m, the walking policy has stood still for
//     dance_still_s and the dance policy has output; after any end, only on a fresh
//     request (the heartbeat must lapse first).
// The dance's clock counts control steps from 0 at the first dancing step; the policy
// runner plays the dance's reference by it.
#pragma once

#include "r1_hw_bridge/reflex_fsm.hpp"

#include <algorithm>
#include <array>
#include <deque>
#include <utility>

namespace r1_hw_bridge
{
namespace reflex
{

enum class Policy { Walk = 0, Dance = 1 };
enum class DanceState { Off = 0, Dancing = 1, Ending = 2 };

inline const char * PolicyName(Policy p) {return p == Policy::Dance ? "dance" : "walk";}
inline const char * DanceStateName(DanceState s)
{
  switch (s) {
    case DanceState::Off: return "OFF";
    case DanceState::Dancing: return "DANCING";
    case DanceState::Ending: return "ENDING";
  }
  return "?";
}

// The same integers as conductor.py's D_* (the golden file holds them).
enum DanceReason : int
{
  kDanceNone = 0, kRefusedDisabled = 1, kRefusedNotArmed = 2, kRefusedStale = 3,
  kRefusedTooClose = 4, kRefusedNotStill = 5, kRefusedNoPolicy = 6, kEndRequest = 7,
  kEndHeartbeat = 8, kEndMax = 9, kEndStale = 10, kStopRetreat = 11, kStopFault = 12,
  kStopOff = 13, kDanceDone = 14,
};

inline const char * DanceReasonText(int r)
{
  switch (r) {
    case kDanceNone: return "";
    case kRefusedDisabled: return "dance refused: dancing (or the reflex) is disabled";
    case kRefusedNotArmed: return "dance refused: the reflex is stepping back or holding the robot";
    case kRefusedStale: return "dance refused: no fresh camera frame";
    case kRefusedTooClose: return "dance refused: someone or something is too close";
    case kRefusedNotStill: return "dance refused: not standing still yet";
    case kRefusedNoPolicy: return "dance refused: the dance policy is not running";
    case kEndRequest: return "dance ending: asked to stop";
    case kEndHeartbeat: return "dance ending: no heartbeat";
    case kEndMax: return "dance ending: maximum length";
    case kEndStale: return "dance ending: camera frames stale";
    case kStopRetreat: return "dance stopped: stepped back";
    case kStopFault: return "dance stopped: dance policy fault";
    case kStopOff: return "dance stopped: bridge not RUNNING or reflex off";
    case kDanceDone: return "danced; back to standing";
  }
  return "?";
}

struct DanceConfig
{
  bool enable = true;
  double person_trigger_m = 0.9;
  double obstacle_trigger_m = 0.9;
  double clear_m = 1.0;
  double still_s = 1.0;
  double heartbeat_s = 0.5;
  double max_s = 63.0;
  std::array<int, 4> bar_ticks{150, 150, 100, 100};   // v3/dance/routine.py BARS at 50 Hz
};

inline constexpr double kDanceDt = 0.02;

// The dance message of one step: none, "dance" (also the heartbeat), or "end".
enum class DanceMsg { None, Dance, End };

struct ConductorOutput
{
  Policy policy = Policy::Walk;
  long dance_tick = -1;
  Output reflex;               // the walking policy's command and the reflex's state
  DanceState dance_state = DanceState::Off;
  int dance_reason = kDanceNone;
};

class Conductor
{
public:
  explicit Conductor(const Config & reflex_cfg = Config(), const DanceConfig & dance_cfg = DanceConfig())
  : fsm_(reflex_cfg), c_(dance_cfg)
  {
    long acc = 0;
    for (std::size_t i = 0; i < c_.bar_ticks.size(); ++i) {
      boundaries_[i] = acc;
      acc += c_.bar_ticks[i];
    }
    loop_ = acc;
  }

  const Fsm & reflex() const {return fsm_;}
  const DanceConfig & dance_config() const {return c_;}
  Policy policy() const {return policy_;}
  long loop_ticks() const {return loop_;}

  bool AtBoundary(long tick) const
  {
    const long m = tick % loop_;
    return std::find(boundaries_.begin(), boundaries_.end(), m) != boundaries_.end();
  }

  ConductorOutput Step(double t, double vx, double vy, double wz, const Frame * frame = nullptr,
                       bool reset = false, bool running = true, DanceMsg dance = DanceMsg::None,
                       bool dance_ok = true)
  {
    if (need_fresh_ && (!have_heartbeat_ || t - t_heartbeat_ > c_.heartbeat_s + 1e-9)) {
      need_fresh_ = false;
    }
    if (dance == DanceMsg::Dance) {
      t_heartbeat_ = t;
      have_heartbeat_ = true;
    }
    const bool dancing = policy_ == Policy::Dance;
    const std::pair<double, double> trig{c_.person_trigger_m, c_.obstacle_trigger_m};
    const Output out = dancing ?
      fsm_.Step(t, 0.0, 0.0, 0.0, frame, reset, running, &trig) :
      fsm_.Step(t, vx, vy, wz, frame, reset, running, nullptr);

    if (dancing) {
      tick_ += 1;
      if (out.state == State::Retreat || out.state == State::Stand) {
        Stop(kStopRetreat);
      } else if (out.state == State::Off) {
        Stop(kStopOff);
      } else if (!dance_ok) {
        Stop(kStopFault);
      } else {
        if (state_ == DanceState::Dancing) {
          if (dance == DanceMsg::End) {
            state_ = DanceState::Ending;
            reason_ = kEndRequest;
          } else if (t - t_heartbeat_ > c_.heartbeat_s + 1e-9) {
            state_ = DanceState::Ending;
            reason_ = kEndHeartbeat;
          } else if (static_cast<double>(tick_) * kDanceDt >= c_.max_s - 1e-9) {
            state_ = DanceState::Ending;
            reason_ = kEndMax;
          } else if (!fsm_.Fresh(t)) {
            state_ = DanceState::Ending;
            reason_ = kEndStale;
          }
        }
        if (state_ == DanceState::Ending && AtBoundary(tick_) && tick_ > 0) {
          Stop(kDanceDone);
        }
      }
    } else if (dance == DanceMsg::Dance && state_ == DanceState::Off && !need_fresh_) {
      const int why = !dance_ok ? kRefusedNoPolicy : StartCheck(t);
      if (why == kDanceNone) {
        policy_ = Policy::Dance;
        state_ = DanceState::Dancing;
        reason_ = kDanceNone;
        tick_ = 0;
      } else {
        reason_ = why;
      }
    }

    // dance steps count as "not standing still": a new dance needs the walking policy
    // to have stood for still_s after the last one
    const bool zero = policy_ == Policy::Walk && out.vx == 0.0 && out.vy == 0.0 && out.wz == 0.0;
    while (!hist_.empty() && !(t - hist_.front().first < c_.still_s + 1e-9)) {hist_.pop_front();}
    hist_.emplace_back(t, zero);

    ConductorOutput o;
    o.policy = policy_;
    o.dance_tick = policy_ == Policy::Dance ? tick_ : -1;
    o.reflex = out;
    o.dance_state = state_;
    o.dance_reason = reason_;
    return o;
  }

private:
  bool Still(double t) const
  {
    if (hist_.empty() || t - hist_.front().first < c_.still_s - 1e-9) {return false;}
    for (const auto & h : hist_) {
      if (t - h.first < c_.still_s + 1e-9 && !h.second) {return false;}
    }
    return true;
  }

  int StartCheck(double t) const
  {
    if (!c_.enable || !fsm_.config().enable) {return kRefusedDisabled;}
    if (fsm_.state() != State::Armed) {return kRefusedNotArmed;}
    if (!fsm_.Fresh(t)) {return kRefusedStale;}
    const auto d = Fsm::Distances(*fsm_.frame());
    if (!fsm_.frame()->depth_ok || std::min(d.first, d.second) <= c_.clear_m) {return kRefusedTooClose;}
    if (!Still(t)) {return kRefusedNotStill;}
    return kDanceNone;
  }

  void Stop(int reason)
  {
    policy_ = Policy::Walk;
    state_ = DanceState::Off;
    reason_ = reason;
    tick_ = -1;
    need_fresh_ = true;
  }

  Fsm fsm_;
  DanceConfig c_;
  std::array<long, 4> boundaries_{};
  long loop_ = 0;
  Policy policy_ = Policy::Walk;
  DanceState state_ = DanceState::Off;
  int reason_ = kDanceNone;
  long tick_ = -1;
  double t_heartbeat_ = 0.0;
  bool have_heartbeat_ = false;
  bool need_fresh_ = false;
  std::deque<std::pair<double, bool>> hist_;     // (t, walking command applied was zero)
};

}  // namespace reflex
}  // namespace r1_hw_bridge
