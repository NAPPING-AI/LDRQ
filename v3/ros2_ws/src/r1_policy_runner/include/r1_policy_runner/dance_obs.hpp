// V3: the dance policy's observation, built in the runner from the bridge's frame and
// the dance's reference. NO ROS types: unit-tested against the Python reference
// (v3/dance/dance_obs.py, which Isaac's own observation was checked against) with
// nothing but a C++17 compiler (test/test_dance_obs.cpp).
//
// The bridge's frame with the dance on is v2's 85 floats and 6 more:
//   [0,3) gyro  [3,6) gravity  [6,9) velocity command  [9,35) joint_pos (rel)
//   [35,61) joint_vel  [61,85) last APPLIED action  [85,89) IMU quaternion w x y z
//   [89] dance tick (-1 when not dancing)  [90] policy (0 walk, 1 dance)
// The dance observation (137) is, in order: the reference's joint positions (26) and
// velocities (26) at the dance tick; the reference pelvis orientation in the robot's
// pelvis frame (6: first two columns of the rotation matrix of inv(q_robot) * q_ref,
// row by row); gyro (3); joint_pos (26); joint_vel (26); last action (24).
//
// Heading: the reference is turned to the robot's heading, taken every step while not
// dancing and frozen at the first dancing step (tick 0). Arithmetic in double, as numpy
// does it, then stored as float.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace r1_policy_runner
{

inline constexpr std::size_t kV2Frame = 85, kDanceFrame = 91, kDanceObs = 137;
inline constexpr std::size_t kJoints = 26, kActions = 24;

// The table v3/dance/dance_obs.py `export` writes: magic, then frames / joints / bodies
// (u32) and fps (f32), then joint_pos, joint_vel (frames x joints) and the pelvis
// quaternion (frames x 4, w x y z), little-endian float32.
class DanceReference
{
public:
  static constexpr char kMagic[] = "R1DANCEREF1\n";

  explicit DanceReference(const std::string & path)
  {
    std::ifstream in(path, std::ios::binary);
    if (!in) {throw std::runtime_error("cannot open dance reference " + path);}
    char magic[sizeof(kMagic) - 1];
    in.read(magic, sizeof(magic));
    if (!in || std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
      throw std::runtime_error(path + " is not a dance reference");
    }
    uint32_t head[3];
    float fps = 0.0f;
    in.read(reinterpret_cast<char *>(head), sizeof(head));
    in.read(reinterpret_cast<char *>(&fps), sizeof(fps));
    frames_ = head[0];
    if (!in || head[1] != kJoints || frames_ == 0) {
      throw std::runtime_error(path + ": unexpected shape");
    }
    fps_ = fps;
    jp_.resize(frames_ * kJoints);
    jv_.resize(frames_ * kJoints);
    q_.resize(frames_ * 4);
    in.read(reinterpret_cast<char *>(jp_.data()), static_cast<std::streamsize>(jp_.size() * 4));
    in.read(reinterpret_cast<char *>(jv_.data()), static_cast<std::streamsize>(jv_.size() * 4));
    in.read(reinterpret_cast<char *>(q_.data()), static_cast<std::streamsize>(q_.size() * 4));
    if (!in) {throw std::runtime_error(path + ": truncated");}
  }

  std::size_t frames() const {return frames_;}
  float fps() const {return fps_;}
  const float * joint_pos(long tick) const {return jp_.data() + Index(tick) * kJoints;}
  const float * joint_vel(long tick) const {return jv_.data() + Index(tick) * kJoints;}
  const float * pelvis_quat(long tick) const {return q_.data() + Index(tick) * 4;}

private:
  std::size_t Index(long tick) const
  {
    const long n = static_cast<long>(frames_);
    return static_cast<std::size_t>(((tick % n) + n) % n);
  }

  std::size_t frames_ = 0;
  float fps_ = 0.0f;
  std::vector<float> jp_, jv_, q_;
};

using Quat = std::array<double, 4>;   // w x y z

inline Quat QMul(const Quat & a, const Quat & b)
{
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
          a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
          a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

inline Quat QConj(const Quat & q) {return {q[0], -q[1], -q[2], -q[3]};}

inline Quat QUnit(const Quat & q)
{
  const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
  return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

inline double Yaw(const Quat & q)
{
  return std::atan2(2.0 * (q[0] * q[3] + q[1] * q[2]), 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3]));
}

inline Quat YawQuat(double yaw) {return {std::cos(yaw / 2.0), 0.0, 0.0, std::sin(yaw / 2.0)};}

class DanceObsBuilder
{
public:
  explicit DanceObsBuilder(const DanceReference & ref) : ref_(ref) {}

  // frame: kDanceFrame floats; out: kDanceObs floats.
  void Build(const float * frame, float * out)
  {
    const Quat q_robot = QUnit({frame[85], frame[86], frame[87], frame[88]});
    long tick = std::lround(frame[89]);
    const bool dancing = frame[90] > 0.5f && tick >= 0;
    if (!dancing || !have_yaw_ || tick == 0) {
      yaw0_ = Yaw(q_robot);
      have_yaw_ = true;
    }
    if (!dancing) {tick = 0;}     // standby: the pose the dance starts from
    const float * q = ref_.pelvis_quat(tick);
    const Quat q_ref = QMul(YawQuat(yaw0_), QUnit({q[0], q[1], q[2], q[3]}));
    const Quat r = QMul(QConj(q_robot), q_ref);
    const double w = r[0], x = r[1], y = r[2], z = r[3];
    // first two columns of the rotation matrix, row by row
    const double m[6] = {1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                         2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                         2 * (x * z - w * y), 2 * (y * z + w * x)};
    std::size_t k = 0;
    const float * jp = ref_.joint_pos(tick);
    const float * jv = ref_.joint_vel(tick);
    for (std::size_t i = 0; i < kJoints; ++i) {out[k++] = jp[i];}
    for (std::size_t i = 0; i < kJoints; ++i) {out[k++] = jv[i];}
    for (double v : m) {out[k++] = static_cast<float>(v);}
    for (std::size_t i = 0; i < 3; ++i) {out[k++] = frame[i];}               // gyro
    for (std::size_t i = 9; i < 61; ++i) {out[k++] = frame[i];}              // jpos, jvel
    for (std::size_t i = 61; i < 85; ++i) {out[k++] = frame[i];}             // last action
  }

  void Reset() {have_yaw_ = false;}

private:
  const DanceReference & ref_;
  double yaw0_ = 0.0;
  bool have_yaw_ = false;
};

}  // namespace r1_policy_runner
