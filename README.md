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

    git clone --recursive https://github.com/VKCOM/torch_mobile.git

To build the Torch with all dependencies for iOS:

    $ ./generate_ios_framework

To build the Torch with all dependencies for Android set NDK_PATH variable and add it to your PATH, then start:

    $ TOOLCHAIN=<android version> ARCH=<architecture> GCC=<gcc version> ./generate_android_jni
