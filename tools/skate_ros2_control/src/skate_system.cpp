// SkateSystem — ros2_control SystemInterface over the skate_ros2 driver
// topics. See the header for the design rationale.

#include "skate_ros2_control/skate_system.hpp"

#include <chrono>
#include <cmath>
#include <limits>

namespace skate_ros2_control
{

namespace
{
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
}  // namespace

SkateSystem::~SkateSystem()
{
  exec_.cancel();
  if (spin_thread_.joinable()) {
    spin_thread_.join();
  }
}

hardware_interface::CallbackReturn SkateSystem::on_init(
  const hardware_interface::HardwareComponentInterfaceParams & params)
{
  if (SystemInterface::on_init(params) != hardware_interface::CallbackReturn::SUCCESS) {
    return hardware_interface::CallbackReturn::ERROR;
  }
  const auto n = info_.joints.size();
  if (n == 0) {
    RCLCPP_ERROR(rclcpp::get_logger("skate_system"),
                 "no joints in the <ros2_control> URDF block");
    return hardware_interface::CallbackReturn::ERROR;
  }
  for (const auto & j : info_.joints) {
    if (j.command_interfaces.size() != 1 ||
        j.command_interfaces[0].name != hardware_interface::HW_IF_POSITION) {
      RCLCPP_ERROR(rclcpp::get_logger("skate_system"),
                   "joint '%s' must have exactly one 'position' command "
                   "interface (the wire is position-controlled)", j.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
    if (j.state_interfaces.empty() ||
        j.state_interfaces[0].name != hardware_interface::HW_IF_POSITION) {
      RCLCPP_ERROR(rclcpp::get_logger("skate_system"),
                   "joint '%s' must list 'position' as its first state "
                   "interface", j.name.c_str());
      return hardware_interface::CallbackReturn::ERROR;
    }
  }

  pos_.assign(n, 0.0);
  vel_.assign(n, 0.0);
  cmd_.assign(n, kNan);
  latest_pos_.assign(n, kNan);
  latest_vel_.assign(n, 0.0);
  for (size_t i = 0; i < n; ++i) {
    index_[info_.joints[i].name] = i;
  }

  auto param = [this](const std::string & key, const std::string & fallback) {
    auto it = info_.hardware_parameters.find(key);
    return it == info_.hardware_parameters.end() ? fallback : it->second;
  };
  state_topic_ = param("state_topic", "/joint_states");
  command_topic_ = param("command_topic", "/skate/joint_position_cmd");

  // A private node on its own executor thread carries the topic traffic, so
  // read()/write() in the controller_manager loop never block on the DDS.
  node_ = rclcpp::Node::make_shared("skate_system_interface");
  pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(command_topic_, 10);
  sub_ = node_->create_subscription<sensor_msgs::msg::JointState>(
    state_topic_, rclcpp::QoS(10),
    [this](sensor_msgs::msg::JointState::ConstSharedPtr msg) { on_state(*msg); });
  exec_.add_node(node_);
  spin_thread_ = std::thread([this] { exec_.spin(); });

  RCLCPP_INFO(node_->get_logger(),
              "skate_system: %zu joints, state <- %s, commands -> %s",
              n, state_topic_.c_str(), command_topic_.c_str());
  return hardware_interface::CallbackReturn::SUCCESS;
}

void SkateSystem::on_state(const sensor_msgs::msg::JointState & msg)
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (size_t k = 0; k < msg.name.size(); ++k) {
    auto it = index_.find(msg.name[k]);
    if (it == index_.end()) {
      continue;                      // legs / head / grippers — not ours
    }
    if (k < msg.position.size()) {
      latest_pos_[it->second] = msg.position[k];
    }
    if (k < msg.velocity.size()) {
      latest_vel_[it->second] = msg.velocity[k];
    }
  }
  got_state_ = true;
}

hardware_interface::CallbackReturn SkateSystem::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // The driver publishes /joint_states only while robot telemetry is LIVE, so
  // waiting here doubles as a link check: refuse to hand joints to a
  // controller before a single real pose has been seen.
  for (int i = 0; i < 100 && !got_state_; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  if (!got_state_) {
    RCLCPP_ERROR(node_->get_logger(),
                 "no %s traffic after 5 s — is skate_driver running and its "
                 "robot link LIVE?", state_topic_.c_str());
    return hardware_interface::CallbackReturn::ERROR;
  }

  std::lock_guard<std::mutex> lock(mtx_);
  pos_ = latest_pos_;
  vel_ = latest_vel_;
  for (size_t i = 0; i < cmd_.size(); ++i) {
    if (std::isnan(cmd_[i])) {
      cmd_[i] = pos_[i];             // arm at the measured pose, never jump
    }
  }
  RCLCPP_INFO(node_->get_logger(), "skate_system active (armed at pose)");
  return hardware_interface::CallbackReturn::SUCCESS;
}

hardware_interface::CallbackReturn SkateSystem::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  // Stop commanding: the driver's command-freshness deadman dampens the
  // robot ~cmd_timeout after the last write(), firmware-style.
  return hardware_interface::CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> SkateSystem::export_state_interfaces()
{
  std::vector<hardware_interface::StateInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    out.emplace_back(info_.joints[i].name,
                     hardware_interface::HW_IF_POSITION, &pos_[i]);
    out.emplace_back(info_.joints[i].name,
                     hardware_interface::HW_IF_VELOCITY, &vel_[i]);
  }
  return out;
}

std::vector<hardware_interface::CommandInterface> SkateSystem::export_command_interfaces()
{
  std::vector<hardware_interface::CommandInterface> out;
  for (size_t i = 0; i < info_.joints.size(); ++i) {
    out.emplace_back(info_.joints[i].name,
                     hardware_interface::HW_IF_POSITION, &cmd_[i]);
  }
  return out;
}

hardware_interface::return_type SkateSystem::read(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  if (got_state_) {
    std::lock_guard<std::mutex> lock(mtx_);
    pos_ = latest_pos_;
    vel_ = latest_vel_;
  }
  return hardware_interface::return_type::OK;
}

hardware_interface::return_type SkateSystem::write(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  // Publish EVERY cycle: the driver's deadman wants a fresh command stream
  // (firmware semantics), and idle controllers legitimately hold pose.
  sensor_msgs::msg::JointState msg;
  msg.header.stamp = node_->now();
  msg.name.reserve(cmd_.size());
  msg.position.reserve(cmd_.size());
  for (size_t i = 0; i < cmd_.size(); ++i) {
    msg.name.push_back(info_.joints[i].name);
    msg.position.push_back(std::isnan(cmd_[i]) ? pos_[i] : cmd_[i]);
  }
  pub_->publish(msg);
  return hardware_interface::return_type::OK;
}

}  // namespace skate_ros2_control

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(
  skate_ros2_control::SkateSystem, hardware_interface::SystemInterface)
