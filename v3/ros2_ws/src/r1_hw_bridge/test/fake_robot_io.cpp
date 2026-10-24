// A stand-in for robot_io.cpp with no Unitree SDK and no robot: a robot standing
// still at its default pose, reported at 500 Hz, and a Send() that goes nowhere.
//
// Built only with -DR1_FAKE_ROBOT_IO=ON (CMakeLists.txt), so the whole bridge node --
// its state machine, the observation it builds, the V3 reflex -- can run on a dev box
// against real ROS topics. It cannot exercise anything about the robot itself: CRC,
// the DDS version split, the slot map against the hardware.
#include "r1_hw_bridge/joint_map.hpp"
#include "r1_hw_bridge/robot_io.hpp"

#include <atomic>
#include <chrono>
#include <thread>

namespace r1_hw_bridge
{

struct RobotIo::Impl
{
  RobotIoConfig cfg;
  std::atomic<bool> running{false};
  std::thread thread;
};

RobotIo::RobotIo(const RobotIoConfig & cfg)
: impl_(std::make_unique<Impl>())
{
  impl_->cfg = cfg;
}

RobotIo::~RobotIo()
{
  impl_->running.store(false);
  if (impl_->thread.joinable()) {impl_->thread.join();}
}

void RobotIo::Start(std::function<void(const RobotState &)> on_state)
{
  impl_->running.store(true);
  impl_->thread = std::thread([this, on_state]() {
      RobotState s;
      s.imu.quat = {1.0f, 0.0f, 0.0f, 0.0f};   // upright
      for (std::size_t j = 0; j < kNumJoints; ++j) {s.motor[kJointSlot[j]].q = kDefaultPos[j];}
      auto next = std::chrono::steady_clock::now();
      while (impl_->running.load()) {
        on_state(s);
        next += std::chrono::microseconds(2000);
        std::this_thread::sleep_until(next);
      }
    });
}

void RobotIo::Send(const std::array<MotorCommand, kSlots> &, uint8_t) {}

std::size_t RobotIo::crc_failures() const {return 0;}

bool RobotIo::output_enabled() const {return false;}

}  // namespace r1_hw_bridge
