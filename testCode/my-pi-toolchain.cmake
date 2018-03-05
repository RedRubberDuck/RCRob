SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

SET(DEVROOT $ENV{HOME}/Workspaces/git/pi/pidev)
SET(PIROOT /home/nandi/Workspaces/git/pi/pidev/piroot )
SET(PITOOLS ${DEVROOT}/pitools)

SET(TOOLROOT ${PITOOLS}/arm-bcm2708/gcc-linaro-4.9-2015.02-3-x86_64_arm-linux-gnueabihf/)

# specify the cross compiler
SET(CMAKE_C_COMPILER   ${TOOLROOT}/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER ${TOOLROOT}/bin/arm-linux-gnueabihf-g++)




SET(CMAKE_SYSROOT ${PIROOT})
SET(CMAKE_FIND_ROOT_PATH ${PIROOT})
# SET(CMAKE_SYSTEM_PREFIX_PATH ${PIROOT})
# SET(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH}  ${PIROOT}/usr/share/cmake-3.6/Modules/)


SET(FLAGS "-Wl,-rpath-link,${PIROOT}/opt/vc/lib -Wl,-rpath-link,${PIROOT}/lib/arm-linux-gnueabihf -Wl,-rpath-link,${PIROOT}/usr/lib/arm-linux-gnueabihf -Wl,-rpath-link,${PIROOT}/usr/local/lib")

UNSET(CMAKE_C_FLAGS CACHE)
UNSET(CMAKE_CXX_FLAGS CACHE)

SET(CMAKE_CXX_FLAGS ${FLAGS} CACHE STRING "" FORCE)
SET(CMAKE_C_FLAGS ${FLAGS} CACHE STRING "" FORCE)


# search for programs in the build host directories
SET(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
# for libraries and headers in the target directories
SET(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
SET(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)


