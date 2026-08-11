#include "../h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void)
{
    char err[256];
    h3_gpu *gpu = h3_gpu_create(NULL, err, sizeof(err));
    if (!gpu) {
        fprintf(stderr, "h3_gpu_create failed: %s\n", err);
        return 1;
    }
    int dev = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    printf("h3 CUDA backend ok\n");
    printf("device: %s\n", prop.name);
    printf("sm: %d.%d\n", prop.major, prop.minor);
    printf("global mem: %.2f GiB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
    /* quick tensor op */
    const size_t n = 1024;
    float *hf = (float*)malloc(n*sizeof(float));
    for (size_t i=0;i<n;i++) hf[i] = (float)i;
    h3_gpu_tensor *t = h3_gpu_tensor_from_f32(gpu, hf, n);
    h3_gpu_tensor *o = h3_gpu_tensor_new_f32(gpu, n);
    if (!t || !o) { fprintf(stderr, "tensor alloc fail: %s\n", h3_gpu_error(gpu)); return 2; }
    if (h3_gpu_begin(gpu) != 0) return 3;
    if (h3_gpu_silu_f32(gpu, o, t, (uint32_t)n) != 0) {
        fprintf(stderr, "silu fail: %s\n", h3_gpu_error(gpu)); return 4;
    }
    if (h3_gpu_submit(gpu) != 0) return 5;
    float *out = (float*)malloc(n*sizeof(float));
    h3_gpu_tensor_read_f32(o, out, n);
    printf("silu(0)=%.6f silu(1)=%.6f silu(2)=%.6f\n", out[0], out[1], out[2]);
    h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(o);
    free(hf); free(out);
    h3_gpu_free(gpu);
    return 0;
}
