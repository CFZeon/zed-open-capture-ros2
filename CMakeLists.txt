cmake_minimum_required(VERSION 3.5)
project(zed_opencv_wrapper)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -Wno-missing-field-initializers -Wno-implicit-fallthrough)
endif()


# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(perception_pcl REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(OpenCV 4.2.0 REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


include_directories(
  include
  ${rclcpp_INCLUDE_DIRS}
  ${rclcpp_action_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
)

set(HEADERS
  include/zed_opencv_wrapper/videocapture.hpp
  include/zed_opencv_wrapper/calibration.hpp
)

add_definitions(-DROS2_BUILD)

link_directories(${catkin_LIBRARY_DIRS})

add_library(videocapture src/videocapture.cpp)

add_executable(zed_pc_node
  src/zed_pc_node.cpp
)

target_link_libraries(zed_pc_node videocapture)

ament_target_dependencies(zed_pc_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  perception_pcl
  OpenCV
)

install(TARGETS
  zed_pc_node
  # camera_calibration
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
