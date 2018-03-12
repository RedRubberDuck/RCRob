

#####################################
# set(CMAKE_TOOLCHAIN_FILE ../my-pi-toolchain.cmake)
cmake_minimum_required (VERSION 3.6) 
project (crossbuild_test)



find_package(OpenCV)

include_directories( / ${OpenCV_INCLUDE_DIRS} ${PIROOT}/usr/local/include ${PIROOT}/usr/include ${PIROOT}/usr/include/python3.4m)
link_directories( ${PIROOT}/opt/vc/lib ${PIROOT}/usr/local/lib ${PIROOT}/usr/lib/python3.4/config-3.4m-arm-linux-gnueabihf/ )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


add_executable (simpletest_hello hello.cpp myPreprocess.cpp)  
target_link_libraries (simpletest_hello ${OpenCV_LIBS}  )


add_executable (test2 test2.cpp )  
target_link_libraries (test2 ${OpenCV_LIBS}  -lpython3.4m  -lraspicam -lraspicam_cv -lmmal -lmmal_core -lmmal_util -lopencv_core -lopencv_highgui)
#####################################