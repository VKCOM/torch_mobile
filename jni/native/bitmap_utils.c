#include "bitmap_utils.h"

int initBitmap(JNIEnv *env, jobject bitmap, AndroidBitmapInfo *infoPointer, rgba **rgbaArrayPointer) {
    if (!bitmap) {
        return BITMAP_ERROR;
    }

    if (AndroidBitmap_getInfo(env, bitmap, infoPointer) < 0) {
        return BITMAP_ERROR;
    }
    if ((*infoPointer).format != ANDROID_BITMAP_FORMAT_RGBA_8888
        || !(*infoPointer).width || !(*infoPointer).height || !(*infoPointer).stride) {
        return BITMAP_ERROR;
    }

    void *pixels = 0;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return BITMAP_ERROR;
    }
    *rgbaArrayPointer = (rgba*) pixels;

    return BITMAP_OK;
}

void releaseBitmap(JNIEnv *env, jobject bitmap) {
    AndroidBitmap_unlockPixels(env, bitmap);
}

