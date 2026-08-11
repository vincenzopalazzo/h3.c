/* Portable RGB24 high-quality resize used on non-Apple platforms.
 * Bicubic (Catmull-Rom) per channel. Metal/macOS keeps Accelerate/vImage.
 */
#include "h3_host.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static float h3_cubic_weight(float x)
{
    x = fabsf(x);
    if (x <= 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    }
    return 0.0f;
}

static uint8_t h3_clamp_u8(float v)
{
    if (v < 0.0f) return 0;
    if (v > 255.0f) return 255;
    return (uint8_t)lrintf(v);
}

static void h3_resize_plane_bicubic(const uint8_t *src, int sw, int sh,
                                   uint8_t *dst, int dw, int dh, int channels)
{
    const float x_ratio = (dw > 1) ? (float)(sw - 1) / (float)(dw - 1) : 0.0f;
    const float y_ratio = (dh > 1) ? (float)(sh - 1) / (float)(dh - 1) : 0.0f;
    for (int y = 0; y < dh; ++y) {
        float sy = y * y_ratio;
        int y0 = (int)floorf(sy);
        float dy = sy - (float)y0;
        for (int x = 0; x < dw; ++x) {
            float sx = x * x_ratio;
            int x0 = (int)floorf(sx);
            float dx = sx - (float)x0;
            for (int c = 0; c < channels; ++c) {
                float acc = 0.0f;
                float wsum = 0.0f;
                for (int m = -1; m <= 2; ++m) {
                    int yy = y0 + m;
                    if (yy < 0) yy = 0;
                    if (yy >= sh) yy = sh - 1;
                    float wy = h3_cubic_weight(dy - (float)m);
                    for (int n = -1; n <= 2; ++n) {
                        int xx = x0 + n;
                        if (xx < 0) xx = 0;
                        if (xx >= sw) xx = sw - 1;
                        float wx = h3_cubic_weight(dx - (float)n);
                        float w = wx * wy;
                        acc += w * (float)src[(size_t)(yy * sw + xx) * (size_t)channels + (size_t)c];
                        wsum += w;
                    }
                }
                dst[(size_t)(y * dw + x) * (size_t)channels + (size_t)c] =
                    h3_clamp_u8(wsum > 0.0f ? acc / wsum : 0.0f);
            }
        }
    }
}

int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 uint8_t *output, int output_width,
                                 int output_height)
{
    if (!input || !output || frames <= 0 ||
        input_width <= 0 || input_height <= 0 ||
        output_width <= 0 || output_height <= 0) {
        return -1;
    }
    if (input_width == output_width && input_height == output_height) {
        size_t n = (size_t)frames * (size_t)input_width * (size_t)input_height * 3u;
        memcpy(output, input, n);
        return 0;
    }
    const size_t src_stride = (size_t)input_width * (size_t)input_height * 3u;
    const size_t dst_stride = (size_t)output_width * (size_t)output_height * 3u;
    for (int f = 0; f < frames; ++f) {
        h3_resize_plane_bicubic(input + (size_t)f * src_stride,
                                input_width, input_height,
                                output + (size_t)f * dst_stride,
                                output_width, output_height, 3);
    }
    return 0;
}
