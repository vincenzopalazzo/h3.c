/* h3 CUDA backend for NVIDIA GPUs (DGX Spark / GB10 first).
 *
 * Implements h3_gpu.h on CUDA 13 + cuBLASLt. Metal remains the default on macOS.
 * Build with: make cuda-spark
 */
#include "h3_gpu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include <float.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct h3_gpu_tensor {
    h3_gpu *gpu;
    void *device;
    size_t elements;
    size_t bytes;
    h3_gpu_dtype dtype;
    int owned;
};

struct h3_gpu {
    int device;
    cudaStream_t stream;
    cublasHandle_t cublas;
    cublasLtHandle_t cublaslt;
    char error[1024];
    int has_error;
    h3_gpu_stats stats;
    char profile_label[128];
    /* staging */
    void *host_stage;
    size_t host_stage_bytes;
    void *d_workspace;
    size_t workspace_bytes;
};

static size_t h3_dtype_size(h3_gpu_dtype dt)
{
    switch (dt) {
    case H3_GPU_F32: return 4;
    case H3_GPU_BF16: return 2;
    case H3_GPU_I8: return 1;
    case H3_GPU_U32: return 4;
    default: return 0;
    }
}

static void h3_set_error(h3_gpu *gpu, const char *fmt, ...)
{
    if (!gpu) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(gpu->error, sizeof(gpu->error), fmt, ap);
    va_end(ap);
    gpu->has_error = 1;
}

static int h3_cuda_ok(h3_gpu *gpu, cudaError_t err, const char *what)
{
    if (err == cudaSuccess) return 1;
    h3_set_error(gpu, "%s: %s", what, cudaGetErrorString(err));
    return 0;
}

static int h3_cublas_ok(h3_gpu *gpu, cublasStatus_t st, const char *what)
{
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    h3_set_error(gpu, "%s: cublas status %d", what, (int)st);
    return 0;
}

static void h3_stats_add_alloc(h3_gpu *gpu, size_t bytes)
{
    if (!gpu) return;
    gpu->stats.allocated_bytes += bytes;
    gpu->stats.live_bytes += bytes;
    if (gpu->stats.live_bytes > gpu->stats.peak_live_bytes)
        gpu->stats.peak_live_bytes = gpu->stats.live_bytes;
    gpu->stats.tensor_allocations += 1;
}

static void h3_stats_sub_alloc(h3_gpu *gpu, size_t bytes)
{
    if (!gpu) return;
    if (gpu->stats.live_bytes >= bytes) gpu->stats.live_bytes -= bytes;
    else gpu->stats.live_bytes = 0;
}

static h3_gpu_tensor *h3_tensor_alloc(h3_gpu *gpu, size_t elements, h3_gpu_dtype dt)
{
    if (!gpu || elements == 0) return NULL;
    size_t es = h3_dtype_size(dt);
    if (!es) return NULL;
    h3_gpu_tensor *t = (h3_gpu_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    t->gpu = gpu;
    t->elements = elements;
    t->bytes = elements * es;
    t->dtype = dt;
    t->owned = 1;
    if (!h3_cuda_ok(gpu, cudaMalloc(&t->device, t->bytes), "cudaMalloc tensor")) {
        free(t);
        return NULL;
    }
    if (!h3_cuda_ok(gpu, cudaMemsetAsync(t->device, 0, t->bytes, gpu->stream), "cudaMemset tensor")) {
        cudaFree(t->device);
        free(t);
        return NULL;
    }
    h3_stats_add_alloc(gpu, t->bytes);
    return t;
}

h3_gpu *h3_gpu_create(const char *shader_source_path, char *error, size_t error_size)
{
    (void)shader_source_path; /* Metal shader path unused on CUDA */
    if (error && error_size) error[0] = '\0';
    h3_gpu *gpu = (h3_gpu *)calloc(1, sizeof(*gpu));
    if (!gpu) {
        if (error && error_size) snprintf(error, error_size, "oom");
        return NULL;
    }
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
        if (error && error_size) snprintf(error, error_size, "no CUDA devices");
        free(gpu);
        return NULL;
    }
    gpu->device = 0;
    if (cudaSetDevice(gpu->device) != cudaSuccess) {
        if (error && error_size) snprintf(error, error_size, "cudaSetDevice failed");
        free(gpu);
        return NULL;
    }
    if (cudaStreamCreateWithFlags(&gpu->stream, cudaStreamNonBlocking) != cudaSuccess) {
        if (error && error_size) snprintf(error, error_size, "cudaStreamCreate failed");
        free(gpu);
        return NULL;
    }
    if (cublasCreate(&gpu->cublas) != CUBLAS_STATUS_SUCCESS) {
        if (error && error_size) snprintf(error, error_size, "cublasCreate failed");
        cudaStreamDestroy(gpu->stream);
        free(gpu);
        return NULL;
    }
    cublasSetStream(gpu->cublas, gpu->stream);
    if (cublasLtCreate(&gpu->cublaslt) != CUBLAS_STATUS_SUCCESS) {
        if (error && error_size) snprintf(error, error_size, "cublasLtCreate failed");
        cublasDestroy(gpu->cublas);
        cudaStreamDestroy(gpu->stream);
        free(gpu);
        return NULL;
    }
    /* default staging 64 MiB */
    gpu->host_stage_bytes = 64ull << 20;
    if (cudaMallocHost(&gpu->host_stage, gpu->host_stage_bytes) != cudaSuccess) {
        gpu->host_stage = NULL;
        gpu->host_stage_bytes = 0;
    }
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "cuda");
    return gpu;
}

void h3_gpu_free(h3_gpu *gpu)
{
    if (!gpu) return;
    if (gpu->d_workspace) cudaFree(gpu->d_workspace);
    if (gpu->host_stage) cudaFreeHost(gpu->host_stage);
    if (gpu->cublaslt) cublasLtDestroy(gpu->cublaslt);
    if (gpu->cublas) cublasDestroy(gpu->cublas);
    if (gpu->stream) cudaStreamDestroy(gpu->stream);
    free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_nax_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_has_int8_mlp(const h3_gpu *gpu) { (void)gpu; return 0; }

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements)
{ return h3_tensor_alloc(gpu, elements, H3_GPU_F32); }
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements)
{ return h3_tensor_alloc(gpu, elements, H3_GPU_BF16); }
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements)
{ return h3_tensor_alloc(gpu, elements, H3_GPU_I8); }

static h3_gpu_tensor *h3_tensor_from_host(h3_gpu *gpu, const void *values, size_t elements,
                                          h3_gpu_dtype dt)
{
    h3_gpu_tensor *t = h3_tensor_alloc(gpu, elements, dt);
    if (!t) return NULL;
    size_t bytes = elements * h3_dtype_size(dt);
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(t->device, values, bytes, cudaMemcpyHostToDevice, gpu->stream),
                    "tensor H2D")) {
        h3_gpu_tensor_free(t);
        return NULL;
    }
    if (!h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "tensor H2D sync")) {
        h3_gpu_tensor_free(t);
        return NULL;
    }
    return t;
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values, size_t elements)
{ return h3_tensor_from_host(gpu, values, elements, H3_GPU_F32); }
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values, size_t elements)
{ return h3_tensor_from_host(gpu, values, elements, H3_GPU_BF16); }
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values, size_t elements)
{ return h3_tensor_from_host(gpu, values, elements, H3_GPU_U32); }

static h3_gpu_tensor *h3_tensor_load_file(h3_gpu *gpu, const char *path, size_t elements, h3_gpu_dtype dt)
{
    if (!gpu || !path) return NULL;
    h3_gpu_tensor *t = h3_tensor_alloc(gpu, elements, dt);
    if (!t) return NULL;
    if (h3_gpu_tensor_read_file_bf16(t, path, 0, elements) != 0 && dt == H3_GPU_BF16) {
        h3_gpu_tensor_free(t);
        return NULL;
    }
    if (dt == H3_GPU_F32) {
        /* generic file read */
        FILE *f = fopen(path, "rb");
        if (!f) { h3_gpu_tensor_free(t); return NULL; }
        size_t bytes = t->bytes;
        size_t off = 0;
        while (off < bytes) {
            size_t chunk = bytes - off;
            if (gpu->host_stage_bytes && chunk > gpu->host_stage_bytes) chunk = gpu->host_stage_bytes;
            void *stage = gpu->host_stage;
            int tmp = 0;
            if (!stage) { stage = malloc(chunk); tmp = 1; if (!stage) { fclose(f); h3_gpu_tensor_free(t); return NULL; } }
            size_t got = fread(stage, 1, chunk, f);
            if (got != chunk) { if (tmp) free(stage); fclose(f); h3_gpu_tensor_free(t); return NULL; }
            if (!h3_cuda_ok(gpu, cudaMemcpyAsync((char*)t->device + off, stage, chunk, cudaMemcpyHostToDevice, gpu->stream), "load f32")) {
                if (tmp) free(stage); fclose(f); h3_gpu_tensor_free(t); return NULL;
            }
            if (tmp) free(stage);
            off += chunk;
        }
        fclose(f);
        cudaStreamSynchronize(gpu->stream);
    }
    return t;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path, size_t elements)
{
    h3_gpu_tensor *t = h3_tensor_alloc(gpu, elements, H3_GPU_BF16);
    if (!t) return NULL;
    if (h3_gpu_tensor_read_file_bf16(t, path, 0, elements) != 0) { h3_gpu_tensor_free(t); return NULL; }
    return t;
}
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path, size_t elements)
{ return h3_tensor_load_file(gpu, path, elements, H3_GPU_F32); }

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements)
{
    if (!tensor || !path || tensor->dtype != H3_GPU_BF16) return -1;
    if (elements > tensor->elements) return -1;
    h3_gpu *gpu = tensor->gpu;
    FILE *f = fopen(path, "rb");
    if (!f) { h3_set_error(gpu, "open %s: %s", path, strerror(errno)); return -1; }
    if (fseeko(f, (off_t)file_offset, SEEK_SET) != 0) { fclose(f); return -1; }
    size_t bytes = elements * 2u;
    size_t off = 0;
    while (off < bytes) {
        size_t chunk = bytes - off;
        if (gpu->host_stage_bytes && chunk > gpu->host_stage_bytes) chunk = gpu->host_stage_bytes;
        void *stage = gpu->host_stage; int tmp=0;
        if (!stage) { stage = malloc(chunk); tmp=1; if(!stage){fclose(f);return -1;} }
        size_t got = fread(stage, 1, chunk, f);
        if (got != chunk) { if(tmp) free(stage); fclose(f); return -1; }
        if (!h3_cuda_ok(gpu, cudaMemcpyAsync((char*)tensor->device + off, stage, chunk, cudaMemcpyHostToDevice, gpu->stream), "read_file_bf16")) {
            if(tmp) free(stage); fclose(f); return -1;
        }
        if (tmp) free(stage);
        off += chunk;
        gpu->stats.blit_copies += 1;
    }
    fclose(f);
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "read_file_bf16 sync") ? 0 : -1;
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements)
{
    /* Same as read for now; true overlapping SSD streaming comes later. */
    return h3_gpu_tensor_read_file_bf16(tensor, path, file_offset, elements);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor)
{
    if (!tensor) return;
    if (tensor->owned && tensor->device) {
        h3_stats_sub_alloc(tensor->gpu, tensor->bytes);
        cudaFree(tensor->device);
    }
    free(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor)
{ return tensor ? tensor->elements : 0; }
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor)
{ return tensor ? tensor->dtype : H3_GPU_F32; }

static int h3_tensor_read(const h3_gpu_tensor *tensor, void *values, size_t elements, h3_gpu_dtype dt)
{
    if (!tensor || !values || tensor->dtype != dt || elements > tensor->elements) return -1;
    size_t bytes = elements * h3_dtype_size(dt);
    h3_gpu *gpu = tensor->gpu;
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(values, tensor->device, bytes, cudaMemcpyDeviceToHost, gpu->stream), "D2H"))
        return -1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "D2H sync") ? 0 : -1;
}
static int h3_tensor_write(h3_gpu_tensor *tensor, const void *values, size_t elements, h3_gpu_dtype dt)
{
    if (!tensor || !values || tensor->dtype != dt || elements > tensor->elements) return -1;
    size_t bytes = elements * h3_dtype_size(dt);
    h3_gpu *gpu = tensor->gpu;
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(tensor->device, values, bytes, cudaMemcpyHostToDevice, gpu->stream), "H2D"))
        return -1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "H2D sync") ? 0 : -1;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values, size_t elements)
{ return h3_tensor_read(tensor, values, elements, H3_GPU_F32); }
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values, size_t elements)
{ return h3_tensor_read(tensor, values, elements, H3_GPU_BF16); }
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values, size_t elements)
{ return h3_tensor_write(tensor, values, elements, H3_GPU_F32); }
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values, size_t elements)
{ return h3_tensor_write(tensor, values, elements, H3_GPU_BF16); }

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor, size_t offset, float *values, size_t elements)
{
    if (!tensor || !values || tensor->dtype != H3_GPU_F32) return -1;
    if (offset + elements > tensor->elements) return -1;
    h3_gpu *gpu = tensor->gpu;
    size_t bytes = elements * 4u;
    const char *src = (const char *)tensor->device + offset * 4u;
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(values, src, bytes, cudaMemcpyDeviceToHost, gpu->stream), "D2H range"))
        return -1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "D2H range sync") ? 0 : -1;
}
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor, size_t offset, const float *values, size_t elements)
{
    if (!tensor || !values || tensor->dtype != H3_GPU_F32) return -1;
    if (offset + elements > tensor->elements) return -1;
    h3_gpu *gpu = tensor->gpu;
    size_t bytes = elements * 4u;
    char *dst = (char *)tensor->device + offset * 4u;
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(dst, values, bytes, cudaMemcpyHostToDevice, gpu->stream), "H2D range"))
        return -1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "H2D range sync") ? 0 : -1;
}
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor, size_t offset, const uint16_t *values, size_t elements)
{
    if (!tensor || !values || tensor->dtype != H3_GPU_BF16) return -1;
    if (offset + elements > tensor->elements) return -1;
    h3_gpu *gpu = tensor->gpu;
    size_t bytes = elements * 2u;
    char *dst = (char *)tensor->device + offset * 2u;
    if (!h3_cuda_ok(gpu, cudaMemcpyAsync(dst, values, bytes, cudaMemcpyHostToDevice, gpu->stream), "H2D bf16 range"))
        return -1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "H2D bf16 range sync") ? 0 : -1;
}

int h3_gpu_begin(h3_gpu *gpu)
{
    if (!gpu) return -1;
    gpu->has_error = 0;
    gpu->error[0] = '\0';
    return 0;
}
int h3_gpu_continue(h3_gpu *gpu)
{
    if (!gpu) return -1;
    if (gpu->has_error) return -1;
    return 0;
}
int h3_gpu_submit(h3_gpu *gpu)
{
    if (!gpu) return -1;
    if (gpu->has_error) return -1;
    gpu->stats.submissions += 1;
    return h3_cuda_ok(gpu, cudaStreamSynchronize(gpu->stream), "submit") ? 0 : -1;
}
const char *h3_gpu_error(const h3_gpu *gpu)
{ return gpu ? gpu->error : "null gpu"; }
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats)
{
    if (!gpu || !stats) return -1;
    *stats = gpu->stats;
    return 0;
}
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label)
{
    if (!gpu) return;
    snprintf(gpu->profile_label, sizeof(gpu->profile_label), "%s", label ? label : "cuda");
}
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase)
{
    (void)gpu; (void)phase;
}

/* ===== kernels ===== */
__global__ void h3_k_copy_f32(float *dst, const float *src, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void h3_k_copy_bf16(__nv_bfloat16 *dst, const __nv_bfloat16 *src, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}
__global__ void h3_k_cast_f32_to_bf16(__nv_bfloat16 *dst, const float *src, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}
__global__ void h3_k_cast_bf16_to_f32(float *dst, const __nv_bfloat16 *src, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __bfloat162float(src[i]);
}
__global__ void h3_k_add_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(a[i]) + __bfloat162float(b[i]);
        out[i] = __float2bfloat16(x);
    }
}
__global__ void h3_k_sub_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *a, const __nv_bfloat16 *b, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(a[i]) - __bfloat162float(b[i]);
        out[i] = __float2bfloat16(x);
    }
}
__global__ void h3_k_silu_f32(float *out, const float *in, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
__global__ void h3_k_silu_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(in[i]);
        float y = x / (1.0f + expf(-x));
        out[i] = __float2bfloat16(y);
    }
}
__global__ void h3_k_gelu_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = __bfloat162float(in[i]);
        /* tanh approximation */
        float y = 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
        out[i] = __float2bfloat16(y);
    }
}
__global__ void h3_k_scale_add_f32(float *out, const float *a, const float *b, float alpha, float beta, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = alpha * a[i] + beta * b[i];
}
__global__ void h3_k_add_scaled_f32(float *out, const float *a, const float *b, float s, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + s * b[i];
}
__global__ void h3_k_clip_f32(float *out, const float *in, float lo, float hi, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = fminf(hi, fmaxf(lo, x));
    }
}
__global__ void h3_k_swiglu_f32(float *out, const float *gate, const float *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float s = g / (1.0f + expf(-g));
        out[i] = s * up[i];
    }
}
__global__ void h3_k_swiglu_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *gate, const __nv_bfloat16 *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float s = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16(s * u);
    }
}
__global__ void h3_k_silu_mul_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *gate, const __nv_bfloat16 *up, size_t n)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float s = g / (1.0f + expf(-g));
        out[i] = __float2bfloat16(s * u);
    }
}
__global__ void h3_k_euler_bf16(float *sample, const __nv_bfloat16 *last, const __nv_bfloat16 *prev,
                               size_t n, float delta, float ratio)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float l = __bfloat162float(last[i]);
        float p = prev ? __bfloat162float(prev[i]) : l;
        float v = l + ratio * (l - p);
        sample[i] += delta * v;
    }
}
__global__ void h3_k_rms_norm_f32(float *out, const float *in, const float *w, uint32_t rows, uint32_t width, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *x = in + (size_t)row * width;
    float *y = out + (size_t)row * width;
    float ms = 0.f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) ms += x[i] * x[i];
    __shared__ float red[256];
    red[threadIdx.x] = ms;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)width + eps);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        float ww = w ? w[i] : 1.f;
        y[i] = x[i] * inv * ww;
    }
}
__global__ void h3_k_rms_norm_bf16(__nv_bfloat16 *out, const __nv_bfloat16 *in, const __nv_bfloat16 *w,
                                  uint32_t rows, uint32_t width, float eps)
{
    uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *x = in + (size_t)row * width;
    __nv_bfloat16 *y = out + (size_t)row * width;
    float ms = 0.f;
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        ms += v*v;
    }
    __shared__ float red[256];
    red[threadIdx.x] = ms;
    __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float inv = rsqrtf(red[0] / (float)width + eps);
    for (uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
        float ww = w ? __bfloat162float(w[i]) : 1.f;
        float v = __bfloat162float(x[i]) * inv * ww;
        y[i] = __float2bfloat16(v);
    }
}

static dim3 h3_grid1d(size_t n, int block=256)
{
    return dim3((unsigned)((n + block - 1) / block));
}

static int h3_need(h3_gpu *gpu, int cond, const char *msg)
{
    if (!cond) { h3_set_error(gpu, "%s", msg); return 0; }
    return 1;
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination, const h3_gpu_tensor *source, uint32_t elements)
{
    if (!gpu||!destination||!source) return -1;
    if (!h3_need(gpu, destination->dtype==H3_GPU_F32 && source->dtype==H3_GPU_F32, "copy_f32 dtype")) return -1;
    h3_k_copy_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)destination->device,(const float*)source->device,elements);
    gpu->stats.direct_dispatches++; gpu->stats.blit_copies++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "copy_f32");
}
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination, const h3_gpu_tensor *source, uint32_t elements)
{
    if (!gpu||!destination||!source) return -1;
    h3_k_copy_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)destination->device,(const __nv_bfloat16*)source->device,elements);
    gpu->stats.direct_dispatches++; gpu->stats.blit_copies++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "copy_bf16") ? 0 : -1;
}
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if (!gpu||!output||!input) return -1;
    h3_k_cast_f32_to_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const float*)input->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "cast_f32_to_bf16") ? 0 : -1;
}
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if (!gpu||!output||!input) return -1;
    h3_k_cast_bf16_to_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const __nv_bfloat16*)input->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "cast_bf16_to_f32") ? 0 : -1;
}
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *a, const h3_gpu_tensor *b, uint32_t elements)
{
    h3_k_add_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)a->device,(const __nv_bfloat16*)b->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "add_bf16") ? 0 : -1;
}
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *a, const h3_gpu_tensor *b, uint32_t elements)
{
    h3_k_sub_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)a->device,(const __nv_bfloat16*)b->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "sub_bf16") ? 0 : -1;
}
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    h3_k_silu_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)input->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "silu_f32") ? 0 : -1;
}
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    h3_k_silu_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "silu_bf16") ? 0 : -1;
}
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    h3_k_gelu_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "gelu_bf16") ? 0 : -1;
}
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *up, uint32_t elements)
{
    h3_k_swiglu_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)gate->device,(const float*)up->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "swiglu_f32") ? 0 : -1;
}
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *up, uint32_t elements)
{
    h3_k_swiglu_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)gate->device,(const __nv_bfloat16*)up->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "swiglu_bf16") ? 0 : -1;
}
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *up, uint32_t elements)
{
    h3_k_silu_mul_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)gate->device,(const __nv_bfloat16*)up->device,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "silu_mul_bf16") ? 0 : -1;
}
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, float lo, float hi, uint32_t elements)
{
    h3_k_clip_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)input->device,lo,hi,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "clip_f32") ? 0 : -1;
}
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *a, const h3_gpu_tensor *b, float alpha, float beta, uint32_t elements)
{
    h3_k_scale_add_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)a->device,(const float*)b->device,alpha,beta,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "scale_add_f32") ? 0 : -1;
}
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *a, const h3_gpu_tensor *b, float scale, uint32_t elements)
{
    h3_k_add_scaled_f32<<<h3_grid1d(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)a->device,(const float*)b->device,scale,elements);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "add_scaled_f32") ? 0 : -1;
}
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, uint32_t rows, uint32_t width, float epsilon)
{
    h3_k_rms_norm_f32<<<rows,256,0,gpu->stream>>>((float*)output->device,(const float*)input->device, weight?(const float*)weight->device:nullptr, rows,width,epsilon);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "rms_norm_f32") ? 0 : -1;
}
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, uint32_t rows, uint32_t width, float epsilon)
{
    h3_k_rms_norm_bf16<<<rows,256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device, weight?(const __nv_bfloat16*)weight->device:nullptr, rows,width,epsilon);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "rms_norm_bf16") ? 0 : -1;
}
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample, size_t sample_offset, const h3_gpu_tensor *last, const h3_gpu_tensor *previous, uint32_t elements, float delta, float ratio)
{
    float *s = (float*)sample->device + sample_offset;
    const __nv_bfloat16 *l = (const __nv_bfloat16*)last->device;
    const __nv_bfloat16 *p = previous ? (const __nv_bfloat16*)previous->device : nullptr;
    h3_k_euler_bf16<<<h3_grid1d(elements),256,0,gpu->stream>>>(s,l,p,elements,delta,ratio);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "euler_bf16") ? 0 : -1;
}

/* Dense linear via cuBLAS: out[M,N] = in[M,K] * w[K,N] + bias[N]
 * Weight layout assumed row-major KxN as common in these engines; if Metal uses
 * transposed layouts, callers still pass leading dims via this API shape.
 */
static int h3_linear_f32_impl(h3_gpu *gpu, float *out, const float *in, const float *w, const float *bias,
                              uint32_t M, uint32_t N, uint32_t K)
{
    const float alpha = 1.f, beta = 0.f;
    /* cublas is column-major: C = A*B with op. We compute out^T = w^T * in^T */
    cublasStatus_t st = cublasSgemm(gpu->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                    (int)N, (int)M, (int)K,
                                    &alpha,
                                    w, (int)N,
                                    in, (int)K,
                                    &beta,
                                    out, (int)N);
    if (!h3_cublas_ok(gpu, st, "cublasSgemm")) return -1;
    if (bias) {
        /* add bias to each row */
        /* simple kernel */
    }
    gpu->stats.mps_linear_dispatches += 1; /* reuse counter field */
    return 0;
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                      uint32_t rows, uint32_t in_features, uint32_t out_features)
{
    if (!gpu||!output||!input||!weight) return -1;
    int rc = h3_linear_f32_impl(gpu, (float*)output->device, (const float*)input->device,
                                (const float*)weight->device,
                                bias?(const float*)bias->device:nullptr,
                                rows, out_features, in_features);
    if (rc != 0) return rc;
    if (bias) {
        /* bias add */
        /* out[m,n] += bias[n] */
        static int warned = 0; (void)warned;
        /* launch tiny kernel via scale by using existing path: copy not enough */
    }
    return 0;
}

/* BF16 linear using cublasGemmEx */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
                       uint32_t rows, uint32_t in_features, uint32_t out_features)
{
    if (!gpu||!output||!input||!weight) return -1;
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmEx(gpu->cublas,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     (int)out_features, (int)rows, (int)in_features,
                                     &alpha,
                                     weight->device, CUDA_R_16BF, (int)out_features,
                                     input->device, CUDA_R_16BF, (int)in_features,
                                     &beta,
                                     output->device, CUDA_R_16BF, (int)out_features,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!h3_cublas_ok(gpu, st, "cublasGemmEx bf16")) return -1;
    gpu->stats.mps_linear_dispatches += 1;
    (void)bias; /* bias epilogue TODO */
    return 0;
}


/* ===== auto stubs for remaining API surface ===== */
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_patch_linear_bf16");
    return -1;
}

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_patch_linear_bf16_offset");
    return -1;
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_patch_linear_bf16_map");
    return -1;
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_adaln_f32");
    return -1;
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_gate_f32");
    return -1;
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_qkv_rope_f32");
    return -1;
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_sdpa_f32");
    return -1;
}

int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_layer_norm_f32");
    return -1;
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_video_qkv_rope_f32");
    return -1;
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t padding, uint32_t dilation)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_conv1d_f32");
    return -1;
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t dilation)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_conv1d_stride_f32");
    return -1;
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_conv_transpose1d_f32");
    return -1;
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *vector, const h3_gpu_tensor *magnitude, uint32_t outer, uint32_t inner)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_weight_norm_f32");
    return -1;
}

int h3_gpu_alias_free_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log, const h3_gpu_tensor *upsample_filter, const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length, uint32_t channels)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_alias_free_snake_f32");
    return -1;
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha, uint32_t batch, uint32_t length, uint32_t channels)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_snake1d_f32");
    return -1;
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias, const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_audio_qkv_split_f32");
    return -1;
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_sdpa_causal_f32");
    return -1;
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_audio_attention_pool_f32");
    return -1;
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *linear, uint32_t elements)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_geglu_f32");
    return -1;
}

int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front, uint32_t height_before, uint32_t height_after, uint32_t width_before, uint32_t width_after)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_vae_encoder_pad_f32");
    return -1;
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_conv3d_f32");
    return -1;
}

int h3_gpu_vae_encoder_group_norm_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t groups, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_vae_encoder_group_norm_silu_f32");
    return -1;
}

int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_mlp_bf16");
    return -1;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_mlp_nax_bf16");
    return -1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows, uint32_t columns)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_quantize_weight_int8");
    return -1;
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim, uint32_t output_dim, int use_slower_uncached_int8_scales)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_linear_int8_bf16");
    return -1;
}

int h3_gpu_linear_int8_head_major_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_linear_int8_head_major_bf16");
    return -1;
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, h3_gpu_tensor *quantized_activation, h3_gpu_tensor *activation_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc1_scales, const h3_gpu_tensor *fc2_weight, const h3_gpu_tensor *fc2_scales, const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim, int use_slower_grouped_quantizer, int use_slower_dynamic_fc1_k, int use_int8_row_fc2, int input_is_quantized)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_mlp_int8_bf16");
    return -1;
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_layer_norm_bf16");
    return -1;
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_vision_qkv_rope_bf16");
    return -1;
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_adaln_bf16");
    return -1;
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_adaln_bf16_offset");
    return -1;
}

int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, uint32_t output_dim, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_adaln_linear_bf16");
    return -1;
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_gate_bf16");
    return -1;
}

int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_gate_adaln_bf16");
    return -1;
}

int h3_gpu_gate_adaln_quantize_int8(h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_gate_adaln_quantize_int8");
    return -1;
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_qkv_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_grouped_qkv_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_grouped_qkv_linear_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_linear_rope_int8(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon, int input_is_quantized, int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms, int use_slower_uncached_int8_scales)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_grouped_qkv_linear_rope_int8");
    return -1;
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_sdpa_bf16");
    return -1;
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_sdpa_bf16_head_major_output");
    return -1;
}

int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *weight, const h3_gpu_tensor *token_ids, uint32_t tokens, uint32_t vocab_size, uint32_t width)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_embedding_bf16");
    return -1;
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output, h3_gpu_tensor *key_output, const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_text_qk_rope_bf16");
    return -1;
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor, const h3_gpu_tensor *weight, uint32_t sequence, uint32_t heads, uint32_t head_dim, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_head_rms_norm_bf16");
    return -1;
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, const h3_gpu_tensor *rope_cos_f32, const h3_gpu_tensor *rope_sin_f32, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_rope_text_bf16");
    return -1;
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_gqa_causal_bf16");
    return -1;
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original, size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs, uint32_t input_rows, uint32_t rows, uint32_t baseline_rows, uint32_t width)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_token_pool_bf16");
    return -1;
}

int h3_gpu_token_pool_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original, size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows, uint32_t baseline_rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_token_pool_adaln_bf16");
    return -1;
}

int h3_gpu_token_expand_delta_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original, size_t original_offset, const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents, uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_token_expand_delta_bf16");
    return -1;
}

int h3_gpu_token_expand_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output, const h3_gpu_tensor *original, size_t original_offset, const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    h3_gpu *gpu_ref = NULL;
    gpu_ref = gpu;
    if (gpu_ref) h3_set_error(gpu_ref, "CUDA stub not implemented: h3_gpu_token_expand_adaln_bf16");
    return -1;
}
