#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>

#include <lauxlib.h>
#include <lua.h>
#include <luaT.h>
#include <TH/TH.h>

#include "rgba.h"
#include "lua_test.h"
#include "bitmap_utils.h"

#include <pthread.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


#define LOG_TAG "TorchNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGV(...) __android_log_print(ANDROID_LOG_VERBOSE, LOG_TAG, __VA_ARGS__)

#define LUA_OK 0
#define LUA_BAD_OPEN -1
#define LUA_WRONG_STATE -2
#define LUA_INIT_SCRIPT_FAILURE -3
#define LUA_USER_SCRIPT_FAILURE -4
#define LUA_SCRIPT_FAILURE -5
#define LUA_ALREADY_EXIST -6
#define LUA_TEST_SCRIPT_ERROR -7
#define BITMAP_LOAD_ERROR -8
#define MAX_STRING_LENGTH 1024

lua_State* state = NULL;

int runTorch(const char* luaPaths, const char* initTorchFilePath, const char* initNNFilePath, const char* mainScriptPath)
{
    LOGI("System call runTorch: OK");
    if (!state)
    {
        lua_executable_dir("./lua");
        state = lua_open();

        LOGI("Run native Torch Lua");

        if (!state)
        {
            return LUA_BAD_OPEN;
        }

        if(lua_status(state))
        {
            return LUA_WRONG_STATE;
        }

        luaL_openlibs(state);
        LOGI("runTorch openlibs: OK");
        lua_getglobal(state, "package");
        lua_getfield(state, -1, "path");
        const char* stdPaths = lua_tostring(state, -1);
        lua_pop(state, 1);

        // add lua files paths to package
        char newPaths[MAX_STRING_LENGTH];
        memset(newPaths, sizeof(char)*MAX_STRING_LENGTH, 0);
        strcat(newPaths, stdPaths);
        strcat(newPaths, ";");
        strcat(newPaths, luaPaths);

        lua_pushstring(state, newPaths);
        lua_setfield(state, -2, "path");
        lua_pop(state, 1);

        // load torch library
        luaopen_libtorch(state);
        LOGI("runTorch load Torch: OK");
        // init torch
        int err = luaL_dofile(state, initTorchFilePath);
        if (err)
        {
            return LUA_INIT_SCRIPT_FAILURE;
        }
        LOGI("runTorch init Torch: OK");

        // load nn library
        luaopen_libnn(state);

        LOGI("runTorch load NN: OK");

        // init nn
        err = luaL_dofile(state, initNNFilePath);
        if (err)
        {
            return LUA_INIT_SCRIPT_FAILURE;
        }

        LOGI("runTorch init NN: OK");
#ifdef USE_BIN_COMPAT
#warning "Torch serializer will work in compatibility mode!"
        // load binary compatibility lib
        luaopen_libbincompat(state);
        LOGI("runTorch init bincompat: OK");
#endif

        // init user scripts
        err = luaL_dofile(state, mainScriptPath);
        if (err)
        {
            return LUA_USER_SCRIPT_FAILURE;
        }

        LOGI("runTorch main.lua: OK");

        //simple lua-script for testing
        err = luaL_dostring(
            state,
            "require 'torch'\n" \
            "require 'nn'\n" \
            "t = torch.FloatTensor(1, 3, 256, 256)\n" \
            "module=nn.SpatialConvolutionMM(3, 32, 3, 3, 1, 1, 0, 0)\n" \
            "o=module:updateOutput(t)\n" \
            "return o:size(3)"
        );

        if(err)
        {
          return LUA_TEST_SCRIPT_ERROR;
        }

        LOGI("runTorch simple NN math: OK");

        // check the size of the output tensor
        if(lua_tointeger(state, -1) == 254)
        {
            return LUA_OK;
        }
        else 
        {
          return LUA_TEST_SCRIPT_ERROR;
        }
    }

    return LUA_ALREADY_EXIST;
}

void freeTorch()
{
    if (state)
    {
        lua_close(state);
        state = NULL;
    }
}

int addVinciFilter(const char* filterPath, int id)
{
    if (state)
    {
        lua_getglobal(state, "loadNeuralNetworks");
        lua_pushstring(state, filterPath);
        lua_pushinteger(state, id);
        int err = lua_pcall(state, 2, 0, 0);
        if (err)
        {
            const char* error_string = lua_tostring(state, -1);
            LOGI(error_string);
            return LUA_SCRIPT_FAILURE;
        }

        return LUA_OK;
    }
    else
    {
        return LUA_WRONG_STATE;
    }
}

int launchVinciFilter(int id, rgba* bitmap, int imageWidth, int imageHeight)
{
    THFloatStorage *input_storage = THFloatStorage_newWithSize4(1, 3, imageHeight, imageWidth);
    THFloatTensor *input = THFloatTensor_newWithStorage4d(input_storage, 0, 1, 3 * imageHeight * imageWidth, 3, imageHeight * imageWidth, imageHeight, imageWidth, imageWidth, 1);
    int x, y;

    for (y = 0; y < imageHeight; y++) {
        for (x = 0; x < imageWidth; x++) {
            int offset = y * imageWidth + x;

            float r = ((float)bitmap[offset].red - 123.939);
            float g = ((float)bitmap[offset].green - 116.779);
            float b = ((float)bitmap[offset].blue - 103.68);

            THTensor_fastSet4d(input, 0, 0, y, x, b);
            THTensor_fastSet4d(input, 0, 1, y, x, g);
            THTensor_fastSet4d(input, 0, 2, y, x, r);
        }
    }

    int resultWidth = imageWidth - 2;
    int resultHeight = imageHeight - 2;

    THFloatStorage *output_storage = THFloatStorage_newWithSize4(1, 3, resultHeight, resultWidth);
    THFloatStorage_fill(output_storage, 0.0f);
    THFloatTensor *output = THFloatTensor_newWithStorage4d(output_storage, 0, 1, 3 * resultHeight * resultWidth, 3, resultHeight * resultWidth, resultHeight, resultWidth, resultWidth, 1);

    lua_getglobal(state, "applyFilter");
    luaT_pushudata(state, input, "torch.FloatTensor");
    luaT_pushudata(state, output, "torch.FloatTensor");
    lua_pushinteger(state, id);

    int ret = lua_pcall(state, 3, 1, 0);

    if (ret)
    {
        const char* error_string = lua_tostring(state, -1);
        LOGI(error_string);

        return LUA_SCRIPT_FAILURE;
    }

    if (!lua_isnumber(state, -1))
    {
        LOGI("user function should return a value");
        return LUA_SCRIPT_FAILURE;
    }

    ret = lua_tonumber(state, -1);
    lua_pop(state, 1);

    if (ret != 42)
    {
        LOGI("something goes wrong in user function");
        return LUA_SCRIPT_FAILURE;
    }

    for (y = 0; y < resultHeight; y++) {
        for (x = 0; x < resultWidth; x++) {
            int offset = y * imageWidth + x;

            float r = THTensor_fastGet4d(output, 0, 2, y, x)*255.0/2 + 255.0/2;
            float g = THTensor_fastGet4d(output, 0, 1, y, x)*255.0/2 + 255.0/2;
            float b = THTensor_fastGet4d(output, 0, 0, y, x)*255.0/2 + 255.0/2;

            r = clampComponent(r);
            g = clampComponent(g);
            b = clampComponent(b);

            bitmap[offset].red = (uint8_t)r;
            bitmap[offset].green = (uint8_t)g;
            bitmap[offset].blue = (uint8_t)b;
        }
    }

    THFloatTensor_free(input);
    THFloatStorage_free(input_storage);
    THFloatTensor_free(output);
    THFloatStorage_free(output_storage);

    return LUA_OK;
}

JNIEXPORT int Java_com_vk_jni_Native_nativeRunTorch(JNIEnv *env, jobject jobj, jstring luaPaths, jstring initTorchFilePath, jstring initNNFilePath, jstring mainScriptPath) {

    const char *nativeLuaPaths = (*env)->GetStringUTFChars(env, luaPaths, 0);
    const char *nativeInitTorchFilePath = (*env)->GetStringUTFChars(env, initTorchFilePath, 0);
    const char *nativeInitNNFilePath = (*env)->GetStringUTFChars(env, initNNFilePath, 0);
    const char *nativeMainScriptPath = (*env)->GetStringUTFChars(env, mainScriptPath, 0);

    int ret = runTorch(nativeLuaPaths, nativeInitTorchFilePath, nativeInitNNFilePath, nativeMainScriptPath);

    (*env)->ReleaseStringUTFChars(env, luaPaths, nativeLuaPaths);
    (*env)->ReleaseStringUTFChars(env, initTorchFilePath, nativeInitTorchFilePath);
    (*env)->ReleaseStringUTFChars(env, initNNFilePath, nativeInitNNFilePath);
    (*env)->ReleaseStringUTFChars(env, mainScriptPath, nativeMainScriptPath);

    return ret;
}

JNIEXPORT int Java_com_vk_jni_Native_nativeAddVinciFilter(JNIEnv *env, jobject jobj, jstring filterPath, int id)
{
    const char *nativeFilterPath = (*env)->GetStringUTFChars(env, filterPath, 0);
    int ret = addVinciFilter(nativeFilterPath, id);
    (*env)->ReleaseStringUTFChars(env, filterPath, nativeFilterPath);

    return ret;
}

JNIEXPORT int Java_com_vk_jni_Native_nativeTorchFilter(JNIEnv *env, jclass class, jobject bitmap, int nid) {
    // Original bitmap

    AndroidBitmapInfo info;
    rgba* input;
    if(initBitmap(env, bitmap, &info, &input) != BITMAP_OK) {
        return BITMAP_LOAD_ERROR;
    }

    int ret = launchVinciFilter(nid, input, info.width, info.height);

    //release resources
    releaseBitmap(env, bitmap);

    return ret;
}
