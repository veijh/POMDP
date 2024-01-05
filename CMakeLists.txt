cmake_minimum_required(VERSION 3.26)
project(cpp_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fopenmp")
#set(MKL_LIBRARIES ${MKL_LP_SEQUENTIAL_LIBRARIES})
set(MKL_LIBRARIES ${MKL_LP_GNUTHREAD_LIBRARIES})

find_package(Eigen3 REQUIRED)
add_definitions(${EIGEN3_DEFINITIONS})
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)
find_package(MKL CONFIG REQUIRED PATHS $ENV{MKLROOT})
include_directories(${MKL_INCLUDE_DIR})


file (GLOB files *.cpp)
foreach (file ${files})
    string(REGEX REPLACE ".+/(.+)\\..*" "\\1" exe ${file})
    add_executable (${exe} ${file} POMDP.cpp)
    message (\ \ \ \ --\ src/${exe}.cpp\ will\ be\ compiled\ to\ bin/${exe})
    target_link_libraries(${exe} ${MKL_LIBRARIES})
endforeach ()
