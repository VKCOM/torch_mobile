Torch7 Library for mobile
=========================

This package has been modified to compile Torch7 for iOS and for Android with the following architectures: armv7, armv7a, arm64.

Code is inherited from the following projects:
* [Torch7 library for iOS by Clement Farabet](https://github.com/clementfarabet/torch-ios)
* [OpenBLAS by Zhang Xianyi](https://github.com/xianyi/OpenBLAS)

Requirements
============

Torch7 has to be installed prior to building the project. A binary tool 'torch' needs to be available in the user's path.

Building
========
To get the code:
`git clone --recursive https://github.com/VKCOM/torch_mobile.git`

To compile and link all dependencies and torch:

iOS:
$ ./generate_ios_framework

Android:
$ NDK_PATH=`path to your android ndk` TOOLCHAIN=`android version` ARCH=`architecture` GCC=`GCC version`./generate_android_jni
