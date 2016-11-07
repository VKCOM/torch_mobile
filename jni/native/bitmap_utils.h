#ifndef BITMAP_UTILS
#define BITMAP_UTILS

#include <android/bitmap.h>
#include "rgba.h"

static const int BITMAP_OK = 0;
static const int BITMAP_ERROR = 1;

int initBitmap(JNIEnv *env, jobject bitmap, AndroidBitmapInfo* infoPointer, rgba** rgbaArrayPointer);

void releaseBitmap(JNIEnv *env, jobject bitmap);

#endif