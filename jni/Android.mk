LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := blas-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libopenblas.a
LOCAL_EXPORT_C_INCLUDES := native/lua
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := torch-lua-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libtorch-lua-static.a
LOCAL_EXPORT_C_INCLUDES := native/lua
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := torch-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libtorch.a
LOCAL_EXPORT_C_INCLUDES := native/lua
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := TH-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libTH.a
LOCAL_EXPORT_C_INCLUDES := native/lua
LOCAL_STATIC_LIBRARIES := blas-prebuild
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := luaT-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libluaT.a
LOCAL_EXPORT_C_INCLUDES := native/lua
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := nn-prebuild
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libnn.a
LOCAL_STATIC_LIBRARIES := TH-prebuild torch-lua-prebuild
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := bincompat
LOCAL_SRC_FILES := libs/$(TARGET_ARCH_ABI)/libbincompat.a
LOCAL_EXPORT_C_INCLUDES := native/lua
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := torchwrapper
LOCAL_SRC_FILES := native/lua_test.c native/bitmap_utils.c
LOCAL_SHARED_LIBRARIES := blas-prebuild torch-prebuild torch-lua-prebuild TH-prebuild luaT-prebuild nn-prebuild
ifeq ($(TARGET_ARCH_ABI),armeabi-v7a-hard)
  LOCAL_ARM_MODE := thumb
  LOCAL_CFLAGS += -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -DUSE_BIN_COMPAT
  LOCAL_SHARED_LIBRARIES += bincompat
  LOCAL_LDFLAGS += -Wl,--no-warn-mismatch -lm_hard
else
  LOCAL_ARM_MODE := arm
endif
LOCAL_LDLIBS := -llog -ljnigraphics
include $(BUILD_SHARED_LIBRARY)

