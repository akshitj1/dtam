cmake_minimum_required(VERSION 2.8)
project( dtam )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake-modules)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# include_directories( ${fmt_INCLUDE_DIRS} )

# set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /EHsc")
add_executable( dtam mapper.cpp)

target_link_libraries( dtam PRIVATE Eigen3::Eigen fmt::fmt ${OpenCV_LIBS})
