cmake_minimum_required(VERSION 3.8)
project(orion_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AudioInfo.msg"
  "msg/AudioData.msg"
  "msg/UserInput.msg"
  "msg/AssistantResponse.msg"
  "msg/TTSRequest.msg"
  "msg/ActionCommand.msg"
  "srv/LLMChat.srv"
  DEPENDENCIES builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
