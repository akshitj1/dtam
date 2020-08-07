cmake_minimum_required(VERSION 2.8)
project( dtam )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( dtam mapper.cpp )
target_link_libraries( dtam ${OpenCV_LIBS} )