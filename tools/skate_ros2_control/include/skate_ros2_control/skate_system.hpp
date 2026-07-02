// Skate ros2_control hardware interface (topic transport).
//
// A hardware_interface::SystemInterface that exposes the Skate's 14 arm
// joints to ros2_control. Transport is deliberately the existing skate_ros2
// driver's topics rather than a second UDP client:
//
//   read()  <-  /joint_states             (driver: calibrated telemetry)
//   write() ->  /skate/joint_position_cmd (driver: by-name JointState)
//
// so ros2_control inherits the driver's arm-at-measured-pose / deadman /
// e-stop / overtemp safety instead of re-implementing it, and the same wire
// code drives sim and hardware. A UDP-native SystemInterface (C++ speaking
// the pickle wire directly) stays the hardware-era option if the extra
// driver hop ever measures as a bottleneck.

#ifndef SKATE_ROS2_CONTROL__SKATE_SYSTEM_HPP_
#define SKATE_ROS2_CONTROL__SKATE_SYSTEM_HPP_

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "hardware_interface/system_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_lifecycle/state.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

namespace skate_ros2_control
{

class SkateSystem : public hardware_interface::SystemInterface
{
public:
  ~SkateSystem() override;

  hardware_interface::CallbackReturn on_init(
    const hardware_interface::HardwareComponentInterfaceParams & params) override;
  hardware_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  hardware_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

  hardware_interface::return_type read(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;
  hardware_interface::return_type write(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void on_state(const sensor_msgs::msg::JointState & msg);

  // per-joint buffers, index-aligned with info_.joints
  std::vector<double> pos_;
  std::vector<double> vel_;
  std::vector<double> cmd_;
  std::unordered_map<std::string, size_t> index_;

  std::string state_topic_;
  std::string command_topic_;

  // background node: the topic bridge to the skate_ros2 driver
  rclcpp::Node::SharedPtr node_;
  rclcpp::executors::SingleThreadedExecutor exec_;
  std::thread spin_thread_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;

  // latest telemetry, written by the executor thread, read by read()
  std::mutex mtx_;
  std::vector<double> latest_pos_;
  std::vector<double> latest_vel_;
  std::atomic<bool> got_state_{false};
};

}  // namespace skate_ros2_control

#endif  // SKATE_ROS2_CONTROL__SKATE_SYSTEM_HPP_
