# -*- cmake -*-
set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
set(CMAKE_CROSSCOMPILING TRUE)

# Project
PROJECT(Torch)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake"
                      "${CMAKE_BINARY_DIR}/cmake" "${CMAKE_MODULE_PATH}")

INCLUDE(TorchPaths)
INCLUDE(TorchPathsInit)
INCLUDE(TorchCFlags)
INCLUDE(TorchPackage)
INCLUDE(TorchWrap)

# Torch executables
ADD_SUBDIRECTORY(exe)

# Torch base packages
ADD_SUBDIRECTORY(pkg)
ADD_SUBDIRECTORY(lib)

# NN Module
ADD_SUBDIRECTORY(nn)

# 32-bit compatibility
ADD_SUBDIRECTORY(bincompat)

# External packages support
INCLUDE(TorchExports)

# Packaging support
INCLUDE(TorchCPack)
