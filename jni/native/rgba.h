#ifndef RGBA
#define RGBA

#include <stdint.h>

#define clampComponent(x) ((x > 255) ? 255 : (x < 0) ? 0 : x)

typedef struct {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
    uint8_t alpha;
} rgba;

#endif