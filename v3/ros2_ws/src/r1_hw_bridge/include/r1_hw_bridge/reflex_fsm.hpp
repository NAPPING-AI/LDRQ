// V3 proximity reflex: the bridge's state machine. NO ROS types, so it builds and is
// tested with nothing but a C++17 compiler (test/test_reflex_fsm.cpp).
//
// It is a line-for-line port of v3/reflex/reflex.py, which is the reference: the
// test replays v3/reflex/golden.csv, every tick of the reference's cases, and the
// outputs here must be the same. Change the Python first, regenerate the golden
// file (v3/reflex/cases.py --write), then change this to match -- in that order, so
// the hand-written checkpoints in cases.py judge the behavior and this file only
// has to agree with it. Arithmetic is written in the Python's order on purpose:
// the comparison is exact.
//
// Once per control step the bridge passes what it would otherwise write into the
// observation's velocity command, the newest proximity frame, a reset request and
// whether it is RUNNING; it writes back the command returned:
//
//   OFF      reflex disabled or the bridge not RUNNING: the command passes.
//   ARMED    the command passes, forward speed clamped to 0 inside the stop
//            distance (it grows with speed) or while the camera cannot vouch for
//            the path (stale frames, no depth, a hole in the depth image's center).
//   RETREAT  (retreat_vx, 0, 0) for retreat_s, whatever the command says.
//   STAND    (0, 0, 0): the policy steps in place. Held until a reset finds the
//            way clear.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace r1_hw_bridge
{
namespace reflex
{

enum class State { Off = 0, Armed = 1, Retreat = 2, Stand = 3 };

inline const char * StateName(State s)
{
  switch (s) {
    case State::Off: return "OFF";
    case State::Armed: return "ARMED";
    case State::Retreat: return "RETREAT";
    case State::Stand: return "STAND";
  }
  return "?";
}

// Reason codes: the same integers as reflex.py's R_* (the golden file holds them).
enum Reason : int
{
  kNone = 0, kDisabled = 1, kNotRunning = 2, kStale = 3, kNoDepth = 4, kDepthHole = 5,
  kStopPerson = 6, kStopObstacle = 7, kRetreatPerson = 8, kRetreatObstacle = 9,
  kStand = 10, kLimit = 11, kResetRefused = 12,
};

inline const char * ReasonText(int r)
{
  switch (r) {
    case kNone: return "";
    case kDisabled: return "reflex disabled";
    case kNotRunning: return "bridge not RUNNING";
    case kStale: return "camera frames stale";
    case kNoDepth: return "no depth";
    case kDepthHole: return "depth hole ahead";
    case kStopPerson: return "person inside the stop distance";
    case kStopObstacle: return "obstacle inside the stop distance";
    case kRetreatPerson: return "stepping back: person too close";
    case kRetreatObstacle: return "stepping back: obstacle too close";
    case kStand: return "standing after stepping back";
    case kLimit: return "stepped back too often -- standing, needs help";
    case kResetRefused: return "reset refused: still too close";
  }
  return "?";
}

// /r1_perception/proximity, std_msgs/Float64MultiArray: the indices of
// v3/perception/r1_perception/pipeline.py PROX_LAYOUT (v3/reflex/test_reflex.py
// checks that the two agree). NaN = no reading.
inline constexpr double kProxVersion = 1.0;
inline constexpr std::size_t kProxVersionIdx = 0, kProxSeq = 1, kProxTCap = 2, kProxTPub = 3,
  kProxPersonM = 4, kProxPersonNearBox = 5, kProxPersons = 6, kProxObstacleM = 7,
  kProxCorridorPoints = 8, kProxDepthValid = 9, kProxCenterValid = 10, kProxDepthOk = 11,
  kProxLen = 12;

inline constexpr double kInf = std::numeric_limits<double>::infinity();

struct Frame
{
  long seq = 0;
  double t_cap = 0.0;                 // CLOCK_MONOTONIC seconds
  std::optional<double> person_m;
  bool person_near_box = false;
  std::optional<double> obstacle_m;
  bool depth_ok = true;
  double center_valid = 1.0;
};

// The message's data -> Frame, or nullopt on a wrong length or version.
inline std::optional<Frame> ParseProximity(const std::vector<double> & d)
{
  if (d.size() != kProxLen || d[kProxVersionIdx] != kProxVersion) {return std::nullopt;}
  auto opt = [](double v) {return std::isnan(v) ? std::optional<double>() : std::optional<double>(v);};
  Frame f;
  f.seq = static_cast<long>(d[kProxSeq]);
  f.t_cap = d[kProxTCap];
  f.person_m = opt(d[kProxPersonM]);
  f.person_near_box = d[kProxPersonNearBox] != 0.0;
  f.obstacle_m = opt(d[kProxObstacleM]);
  f.depth_ok = d[kProxDepthOk] != 0.0;
  f.center_valid = std::isnan(d[kProxCenterValid]) ? 0.0 : d[kProxCenterValid];
  return f;
}

struct Config
{
  bool enable = true;
  double person_trigger_m = 0.6;
  double obstacle_trigger_m = 0.45;
  double clear_margin_m = 0.2;
  int confirm_frames = 2;
  double retreat_vx = -0.3;
  double retreat_s = 1.5;
  int max_retreats = 2;
  double window_s = 10.0;
  double stale_s = 0.3;
  double center_valid_min = 0.2;
  double release_margin_m = 0.10;
  // stop distance (v3/reflex/stop_distance.py)
  double person_floor_m = 1.2;
  double obstacle_floor_m = 0.6;
  double stop_margin_m = 0.10;
  double latency_s = 0.25;
  double k_lat = 1.5;
  double k_brake = 1.3;
  double speed_window_s = 1.0;
};

// (forward command m/s, worst braking distance m): stop_distance.py BRAKE_WORST,
// the v2 policy in simulation under the training randomization, worst of 5 x 256.
inline constexpr double kBrakeV[] = {0.0, 0.3, 0.5, 1.0};
inline constexpr double kBrakeD[] = {0.068, 0.157, 0.228, 0.470};
inline constexpr std::size_t kBrakeN = 4;

inline double Brake(double v)
{
  v = std::max(0.0, v);
  for (std::size_t i = 0; i + 1 < kBrakeN; ++i) {
    if (v <= kBrakeV[i + 1]) {
      return kBrakeD[i] + (kBrakeD[i + 1] - kBrakeD[i]) * (v - kBrakeV[i]) /
             (kBrakeV[i + 1] - kBrakeV[i]);
    }
  }
  const std::size_t a = kBrakeN - 2, b = kBrakeN - 1;
  return kBrakeD[b] + (kBrakeD[b] - kBrakeD[a]) * (v - kBrakeV[b]) / (kBrakeV[b] - kBrakeV[a]);
}

// stop(v) = max(floor, trigger + margin + k_lat * v * latency + k_brake * brake(v))
inline double StopDistance(double trigger, double floor_m, double v, const Config & c)
{
  v = std::max(0.0, v);
  const double need = trigger + c.stop_margin_m + c.k_lat * v * c.latency_s + c.k_brake * Brake(v);
  return std::max(floor_m, need);
}

struct Output
{
  double vx = 0.0, vy = 0.0, wz = 0.0;
  State state = State::Off;
  int reason = kNone;
  bool blocked = false;
  bool reset_refused = false;
};

class Fsm
{
public:
  explicit Fsm(const Config & cfg = Config()) : c_(cfg) {}

  const Config & config() const {return c_;}
  State state() const {return state_;}
  const std::optional<Frame> & frame() const {return frame_;}
  // Retreats started within the window (the newest last), for logging.
  const std::deque<double> & retreats() const {return retreats_;}

  // `triggers` (person m, obstacle m) replaces the retreat triggers for a frame that
  // arrives on this step: the dance (conductor.hpp) triggers farther out.
  Output Step(double t, double vx, double vy, double wz, const Frame * frame = nullptr,
              bool reset = false, bool running = true,
              const std::pair<double, double> * triggers = nullptr)
  {
    if (!c_.enable || !running) {
      state_ = State::Off;
      near_count_ = 0;
      blocked_ = kNone;
      frame_.reset();
      have_seq_ = false;
      return Make(vx, vy, wz, State::Off, !c_.enable ? kDisabled : kNotRunning);
    }
    if (state_ == State::Off) {
      state_ = State::Armed;
      near_count_ = 0;
      rearmed_ = false;
    }

    if (frame != nullptr && (!have_seq_ || frame->seq != last_seq_)) {
      frame_ = *frame;
      last_seq_ = frame->seq;
      have_seq_ = true;
      const bool fresh_frame = t - frame->t_cap <= c_.stale_s;
      const int kind = fresh_frame ? Near(*frame, triggers) : kNoKind;
      // consecutive near frames, person or obstacle alike; the latest names the cause
      if (kind != kNoKind) {
        near_count_ += 1;
        near_kind_ = kind;
      } else {
        near_count_ = 0;
        near_kind_ = kNoKind;
      }
      if (state_ == State::Stand && fresh_frame && ClearOfTrigger(*frame)) {
        rearmed_ = true;
      }
    }

    bool refused = false;
    if (state_ == State::Retreat && t - t_retreat_ >= c_.retreat_s - 1e-9) {
      state_ = State::Stand;
      stand_reason_ = kStand;
      near_count_ = 0;
    }
    if (state_ == State::Stand && reset) {
      if (Fresh(t) && ClearOfTrigger(*frame_)) {
        state_ = State::Armed;
        near_count_ = 0;
        blocked_ = kNone;
      } else {
        refused = true;
      }
    }
    // in STAND only a new approach, and never once the limit has latched
    if (near_count_ >= c_.confirm_frames &&
      (state_ == State::Armed ||
      (state_ == State::Stand && rearmed_ && stand_reason_ != kLimit)))
    {
      StartRetreat(t, near_kind_);
    }

    Output out;
    if (state_ == State::Retreat) {
      out = Make(c_.retreat_vx, 0.0, 0.0, State::Retreat, retreat_reason_);
      blocked_ = kNone;
    } else if (state_ == State::Stand) {
      out = Make(0.0, 0.0, 0.0, State::Stand, refused ? kResetRefused : stand_reason_);
      out.reset_refused = refused;
      blocked_ = kNone;
    } else {
      const int why = BlockReason(t, vx);
      blocked_ = why;
      if (why != kNone && vx > 0.0) {
        out = Make(0.0, vy, wz, State::Armed, why);
      } else {
        out = Make(vx, vy, wz, State::Armed, why);
      }
      out.blocked = why != kNone;
    }
    while (!vx_hist_.empty() && !(t - vx_hist_.front().first < c_.speed_window_s)) {
      vx_hist_.pop_front();
    }
    vx_hist_.emplace_back(t, out.vx);
    return out;
  }

  // The stop distances at forward speed v, for logging.
  double StopPerson(double v) const {return StopDistance(c_.person_trigger_m, c_.person_floor_m, v, c_);}
  double StopObstacle(double v) const
  {
    return StopDistance(c_.obstacle_trigger_m, c_.obstacle_floor_m, v, c_);
  }

  // (person distance, obstacle distance); infinity = nothing seen. A person box tall
  // enough to be near counts as distance 0.
  static std::pair<double, double> Distances(const Frame & f)
  {
    const double p = f.person_near_box ? 0.0 : (f.person_m ? *f.person_m : kInf);
    const double o = (!f.depth_ok || !f.obstacle_m) ? kInf : *f.obstacle_m;
    return {p, o};
  }

  bool Fresh(double t) const {return frame_.has_value() && t - frame_->t_cap <= c_.stale_s;}

private:
  static constexpr int kNoKind = 0, kPerson = 1, kObstacle = 2;

  static Output Make(double vx, double vy, double wz, State s, int reason)
  {
    Output o;
    o.vx = vx;
    o.vy = vy;
    o.wz = wz;
    o.state = s;
    o.reason = reason;
    return o;
  }

  int Near(const Frame & f, const std::pair<double, double> * triggers) const
  {
    const auto d = Distances(f);
    const double tp = triggers ? triggers->first : c_.person_trigger_m;
    const double to = triggers ? triggers->second : c_.obstacle_trigger_m;
    if (d.first < tp) {return kPerson;}
    if (d.second < to) {return kObstacle;}
    return kNoKind;
  }

  bool ClearOfTrigger(const Frame & f) const
  {
    const auto d = Distances(f);
    const double m = c_.clear_margin_m;
    return f.depth_ok && d.first > c_.person_trigger_m + m && d.second > c_.obstacle_trigger_m + m;
  }

  double Speed(double t, double vx_req) const
  {
    double v = vx_req;
    for (const auto & h : vx_hist_) {
      if (t - h.first < c_.speed_window_s) {v = std::max(v, h.second);}
    }
    return std::max(0.0, v);
  }

  int BlockReason(double t, double vx_req) const
  {
    if (!Fresh(t)) {return kStale;}
    const Frame & f = *frame_;
    if (!f.depth_ok) {return kNoDepth;}
    if (f.center_valid < c_.center_valid_min) {return kDepthHole;}
    const double v = Speed(t, vx_req);
    const auto d = Distances(f);
    const double sp = StopPerson(v);
    const double so = StopObstacle(v);
    const double hold_p = blocked_ == kStopPerson ? c_.release_margin_m : 0.0;
    const double hold_o = blocked_ == kStopObstacle ? c_.release_margin_m : 0.0;
    if (d.first < sp + hold_p) {return kStopPerson;}
    if (d.second < so + hold_o) {return kStopObstacle;}
    return kNone;
  }

  void StartRetreat(double t, int kind)
  {
    while (!retreats_.empty() && !(t - retreats_.front() < c_.window_s)) {retreats_.pop_front();}
    near_count_ = 0;
    rearmed_ = false;
    if (static_cast<int>(retreats_.size()) >= c_.max_retreats) {
      state_ = State::Stand;
      stand_reason_ = kLimit;
      return;
    }
    retreats_.push_back(t);
    state_ = State::Retreat;
    t_retreat_ = t;
    retreat_reason_ = kind == kPerson ? kRetreatPerson : kRetreatObstacle;
  }

  Config c_;
  State state_ = State::Off;
  std::optional<Frame> frame_;
  int near_count_ = 0;
  int near_kind_ = kNoKind;
  long last_seq_ = 0;
  bool have_seq_ = false;
  bool rearmed_ = false;
  double t_retreat_ = 0.0;
  std::deque<double> retreats_;
  int stand_reason_ = kStand;
  int retreat_reason_ = kRetreatPerson;
  int blocked_ = kNone;
  std::deque<std::pair<double, double>> vx_hist_;
};

}  // namespace reflex
}  // namespace r1_hw_bridge
