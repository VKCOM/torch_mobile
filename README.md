Torch7 Library for iOS
======================

This package has been modified to compile
Torch7 for iOS (iPad/iPhone) and for Android with the following architectures: armv7, armv7a, arm64

Requirements
============

Torch7 needs to be installed prior to building the project. A binary tool 'torch' needs to be available in the user's path.

I recommend doing the easy install if you have not installed Torch7.
http://torch.ch/docs/getting-started.html

Building The Framework
============
Simply run:
$ ./generate_ios_framework
$ ./generate_android_jni

This will build all torch's libraries as static libs, and export them
in the dirs: framework/ and jni/jniLibs. The dirs are ready to be included in
an iOS or Android project: they includes an example class to load Torch from within
your Objective C or Java project.

For examples full examples that utilize the classes please see 
the examples/ folder. More examples to come soon.

Running
=======
