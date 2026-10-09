cmake_minimum_required(VERSION 3.16)
project(laya_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# MLA buffers, ELF loading and the HF tokenizer come from the LLiMa runtime (sima-lmm-dev).
find_package(SimaLMM REQUIRED)
find_package(nlohmann_json QUIET)

add_library(laya_runtime STATIC src/runtime.cpp src/sequence.cpp src/clm.cpp src/d1.cpp)
target_include_directories(laya_runtime PUBLIC include)
# sima_lmm/mla_model.hpp pulls in the dispatcher headers, which include <json/json.h>.
target_include_directories(laya_runtime PRIVATE /usr/include/jsoncpp)
target_link_libraries(laya_runtime PUBLIC SimaLMM::sima_lmm_runtime)
if(nlohmann_json_FOUND)
  target_link_libraries(laya_runtime PUBLIC nlohmann_json::nlohmann_json)
endif()

# d1 decodes the pictures it is asked about with OpenCV, which the DevKit image has. Without
# it the runtime still builds, and d1 takes pictures as raw pixels only.
find_package(OpenCV QUIET COMPONENTS core imgcodecs)
if(OpenCV_FOUND)
  target_compile_definitions(laya_runtime PRIVATE LAYA_HAVE_OPENCV)
  target_include_directories(laya_runtime PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(laya_runtime PUBLIC ${OpenCV_LIBS})
endif()

add_executable(laya src/main.cpp)
target_link_libraries(laya PRIVATE laya_runtime)
install(TARGETS laya RUNTIME DESTINATION bin)
