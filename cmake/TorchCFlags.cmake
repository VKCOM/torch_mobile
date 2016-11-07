INCLUDE(CheckCCompilerFlag)
INCLUDE(CheckCXXCompilerFlag)

# We want release compilation by default
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

# we want warnings
# we want exceptions support even when compiling c code

# C
CHECK_C_COMPILER_FLAG(-Wall C_HAS_WALL)
IF(C_HAS_WALL)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
ENDIF(C_HAS_WALL)

CHECK_C_COMPILER_FLAG(-Wno-unused-function C_HAS_NO_UNUSED_FUNCTION)
IF(C_HAS_NO_UNUSED_FUNCTION)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
ENDIF(C_HAS_NO_UNUSED_FUNCTION)

CHECK_C_COMPILER_FLAG(-fexceptions C_HAS_FEXCEPTIONS)
IF(C_HAS_FEXCEPTIONS)
  SET(CMAKE_C_FLAGS "-fexceptions ${CMAKE_C_FLAGS}")
ENDIF(C_HAS_FEXCEPTIONS)

# C++
CHECK_CXX_COMPILER_FLAG(-Wall CXX_HAS_WALL)
IF(CXX_HAS_WALL)
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
ENDIF(CXX_HAS_WALL)

CHECK_CXX_COMPILER_FLAG(-Wno-unused-function CXX_HAS_NO_UNUSED_FUNCTION)
IF(CXX_HAS_NO_UNUSED_FUNCTION)
  SET(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-function")
ENDIF(CXX_HAS_NO_UNUSED_FUNCTION)

# When using MSVC
IF(MSVC)
  # we want to respect the standard, and we are bored of those **** .
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE=1)
ENDIF(MSVC)

# OpenMP support?
#SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?")
#IF (APPLE AND CMAKE_COMPILER_IS_GNUCC)
#  EXEC_PROGRAM (uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
#  STRING (REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
#  MESSAGE (STATUS "MAC OS Darwin Version: ${DARWIN_VERSION}")
#  IF (DARWIN_VERSION GREATER 9)
#    SET(APPLE_OPENMP_SUCKS 1)
#  ENDIF (DARWIN_VERSION GREATER 9)
#  EXECUTE_PROCESS (COMMAND ${CMAKE_C_COMPILER} -dumpversion
#    OUTPUT_VARIABLE GCC_VERSION)
#  IF (APPLE_OPENMP_SUCKS AND GCC_VERSION VERSION_LESS 4.6.2)
#    MESSAGE(STATUS "Warning: Disabling OpenMP (unstable with this version of GCC)")
#    MESSAGE(STATUS " Install GCC >= 4.6.2 or change your OS to enable OpenMP")
#    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unknown-pragmas")
#    SET(WITH_OPENMP OFF CACHE BOOL "OpenMP support if available?" FORCE)
#  ENDIF ()
#ENDIF ()

#IF (WITH_OPENMP)
#  FIND_PACKAGE(OpenMP)
#  IF(OPENMP_FOUND)
#    MESSAGE(STATUS "Compiling with OpenMP support")
#    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#  ENDIF(OPENMP_FOUND)
#ENDIF (WITH_OPENMP)

# ARM specific flags
FIND_PACKAGE(ARM)
IF (NEON_FOUND)
  MESSAGE(STATUS "Neon found with compiler flag : -mfpu=neon -D__NEON__")
  SET(CMAKE_C_FLAGS "-mfpu=neon -D__NEON__ ${CMAKE_C_FLAGS}")
ENDIF (NEON_FOUND)
IF (CORTEXA8_FOUND)
  MESSAGE(STATUS "Cortex-A8 Found with compiler flag : -mcpu=cortex-a8")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a8 -fprefetch-loop-arrays ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA8_FOUND)
IF (CORTEXA9_FOUND)
  MESSAGE(STATUS "Cortex-A9 Found with compiler flag : -mcpu=cortex-a9")
  SET(CMAKE_C_FLAGS "-mcpu=cortex-a9 ${CMAKE_C_FLAGS}")
ENDIF (CORTEXA9_FOUND)
