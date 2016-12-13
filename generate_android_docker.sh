. ~/torch/install/bin/torch-activate

TOOLCHAIN=21 ARCH=ARMV8 GCC=4.9 ./generate_android_jni
mkdir -p /workspace/mount/arm64-v8a
cp /workspace/jni/jniLibs/arm64-v8a/arm64-v8a/libtorchwrapper.so /workspace/mount/arm64-v8a/

TOOLCHAIN=19 ARCH=ARMV7 GCC=4.9 ./generate_android_jni
mkdir -p /workspace/mount/armeabi-v7a
cp /workspace/jni/jniLibs/armeabi-v7a-hard/armeabi-v7a/libtorchwrapper.so /workspace/mount/armeabi-v7a/
