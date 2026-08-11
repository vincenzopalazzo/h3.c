#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include "h3_gpu.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct h3_gpu_tensor { h3_gpu *gpu; void *device; size_t elements; size_t bytes; h3_gpu_dtype dtype; int owned; };
struct h3_gpu { int device; cudaStream_t stream; cublasHandle_t cublas; cublasLtHandle_t cublaslt; char error[1024]; int has_error; h3_gpu_stats stats; char profile_label[128]; void *host_stage; size_t host_stage_bytes; };
static size_t h3_dtype_size(h3_gpu_dtype dt){ switch(dt){ case H3_GPU_F32:return 4; case H3_GPU_BF16:return 2; case H3_GPU_I8:return 1; case H3_GPU_U32:return 4; default:return 0; } }
static void h3_set_error(h3_gpu *gpu,const char *fmt,...){ if(!gpu) return; va_list ap; va_start(ap,fmt); vsnprintf(gpu->error,sizeof(gpu->error),fmt,ap); va_end(ap); gpu->has_error=1; }
static int h3_cuda_ok(h3_gpu *gpu,cudaError_t err,const char *what){ if(err==cudaSuccess) return 1; h3_set_error(gpu,"%s: %s",what,cudaGetErrorString(err)); return 0; }
static int h3_cublas_ok(h3_gpu *gpu,cublasStatus_t st,const char *what){ if(st==CUBLAS_STATUS_SUCCESS) return 1; h3_set_error(gpu,"%s: cublas %d",what,(int)st); return 0; }
static void h3_stats_add(h3_gpu *gpu,size_t bytes){ gpu->stats.allocated_bytes+=bytes; gpu->stats.live_bytes+=bytes; if(gpu->stats.live_bytes>gpu->stats.peak_live_bytes) gpu->stats.peak_live_bytes=gpu->stats.live_bytes; gpu->stats.tensor_allocations++; }
static void h3_stats_sub(h3_gpu *gpu,size_t bytes){ if(gpu->stats.live_bytes>=bytes) gpu->stats.live_bytes-=bytes; else gpu->stats.live_bytes=0; }
static h3_gpu_tensor *h3_tensor_alloc(h3_gpu *gpu,size_t elements,h3_gpu_dtype dt){ if(!gpu||!elements) return NULL; size_t es=h3_dtype_size(dt); if(!es) return NULL; h3_gpu_tensor *t=(h3_gpu_tensor*)calloc(1,sizeof(*t)); if(!t) return NULL; t->gpu=gpu; t->elements=elements; t->bytes=elements*es; t->dtype=dt; t->owned=1; if(!h3_cuda_ok(gpu,cudaMalloc(&t->device,t->bytes),"cudaMalloc")){ free(t); return NULL;} if(!h3_cuda_ok(gpu,cudaMemsetAsync(t->device,0,t->bytes,gpu->stream),"cudaMemset")){ cudaFree(t->device); free(t); return NULL;} h3_stats_add(gpu,t->bytes); return t; }
static unsigned h3_blocks(size_t n){ return (unsigned)((n+255ull)/256ull); }
__global__ void h3_k_cast_f32_to_bf16(__nv_bfloat16 *dst,const float *src,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__float2bfloat16(src[i]); }
__global__ void h3_k_cast_bf16_to_f32(float *dst,const __nv_bfloat16 *src,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) dst[i]=__bfloat162float(src[i]); }
__global__ void h3_k_add_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *a,const __nv_bfloat16 *b,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2bfloat16(__bfloat162float(a[i])+__bfloat162float(b[i])); }
__global__ void h3_k_sub_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *a,const __nv_bfloat16 *b,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=__float2bfloat16(__bfloat162float(a[i])-__bfloat162float(b[i])); }
__global__ void h3_k_silu_f32(float *out,const float *in,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float x=in[i]; out[i]=x/(1.f+expf(-x)); } }
__global__ void h3_k_silu_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *in,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float x=__bfloat162float(in[i]); out[i]=__float2bfloat16(x/(1.f+expf(-x))); } }
__global__ void h3_k_gelu_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *in,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float x=__bfloat162float(in[i]); float y=0.5f*x*(1.f+tanhf(0.7978845608f*(x+0.044715f*x*x*x))); out[i]=__float2bfloat16(y);} }
__global__ void h3_k_swiglu_f32(float *out,const float *gate,const float *up,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float g=gate[i]; out[i]=(g/(1.f+expf(-g)))*up[i]; } }
__global__ void h3_k_swiglu_fused_f32(float *out,const float *fused,uint32_t rows,uint32_t width){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t n=(size_t)rows*width; if(i>=n) return;
    size_t row=i/width; size_t col=i%width; const float *x=fused + row*(size_t)width*2u; float g=x[col]; float u=x[col+width]; out[i]=(g/(1.f+expf(-g)))*u;
}
__global__ void h3_k_swiglu_fused_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *fused,uint32_t rows,uint32_t width){
    size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t n=(size_t)rows*width; if(i>=n) return;
    size_t row=i/width; size_t col=i%width; const __nv_bfloat16 *x=fused + row*(size_t)width*2u; float g=__bfloat162float(x[col]); float u=__bfloat162float(x[col+width]); out[i]=__float2bfloat16((g/(1.f+expf(-g)))*u);
}
__global__ void h3_k_swiglu_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *gate,const __nv_bfloat16 *up,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float g=__bfloat162float(gate[i]); float u=__bfloat162float(up[i]); out[i]=__float2bfloat16((g/(1.f+expf(-g)))*u);} }
__global__ void h3_k_clip_f32(float *out,const float *in,float lo,float hi,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float x=in[i]; out[i]=fminf(hi,fmaxf(lo,x)); } }
__global__ void h3_k_scale_add_tensor_f32(float *out,const float *residual,const float *branch,const float *scale,uint32_t rows,uint32_t width){
 size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t n=(size_t)rows*width; if(i>=n) return;
 size_t col=i%width; float s=scale?scale[col]:1.f; out[i]=residual[i]+s*branch[i];
}
__global__ void h3_k_scale_add_f32(float *out,const float *a,const float *b,float alpha,float beta,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=alpha*a[i]+beta*b[i]; }
__global__ void h3_k_add_scaled_f32(float *out,const float *a,const float *b,float sa,float sb,size_t n){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=sa*a[i]+sb*b[i]; }
__global__ void h3_k_rms_norm_f32(float *out,const float *in,const float *w,uint32_t rows,uint32_t width,float eps){ uint32_t row=blockIdx.x; if(row>=rows) return; const float *x=in+(size_t)row*width; float *y=out+(size_t)row*width; float ms=0.f; for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x) ms+=x[i]*x[i]; __shared__ float red[256]; red[threadIdx.x]=ms; __syncthreads(); for(int s=blockDim.x/2;s>0;s>>=1){ if((int)threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); } float inv=rsqrtf(red[0]/(float)width+eps); for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x){ float ww=w?w[i]:1.f; y[i]=x[i]*inv*ww; } }
__global__ void h3_k_rms_norm_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *in,const __nv_bfloat16 *w,uint32_t rows,uint32_t width,float eps){ uint32_t row=blockIdx.x; if(row>=rows) return; const __nv_bfloat16 *x=in+(size_t)row*width; __nv_bfloat16 *y=out+(size_t)row*width; float ms=0.f; for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x){ float v=__bfloat162float(x[i]); ms+=v*v;} __shared__ float red[256]; red[threadIdx.x]=ms; __syncthreads(); for(int s=blockDim.x/2;s>0;s>>=1){ if((int)threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); } float inv=rsqrtf(red[0]/(float)width+eps); for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x){ float ww=w?__bfloat162float(w[i]):1.f; y[i]=__float2bfloat16(__bfloat162float(x[i])*inv*ww);} }
__global__ void h3_k_euler_bf16(float *sample,const __nv_bfloat16 *last,const __nv_bfloat16 *prev,size_t n,float delta,float ratio){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float l=__bfloat162float(last[i]); float p=prev?__bfloat162float(prev[i]):l; sample[i]+=delta*(l+ratio*(l-p)); } }
__global__ void h3_k_bias_add_f32(float *out,const float *bias,uint32_t rows,uint32_t cols){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t n=(size_t)rows*cols; if(i<n) out[i]+=bias[i%cols]; }
__global__ void h3_k_bias_add_bf16(__nv_bfloat16 *out,const __nv_bfloat16 *bias,uint32_t rows,uint32_t cols){ size_t i=blockIdx.x*blockDim.x+threadIdx.x; size_t n=(size_t)rows*cols; if(i<n) out[i]=__float2bfloat16(__bfloat162float(out[i])+__bfloat162float(bias[i%cols])); }

h3_gpu * h3_gpu_create(const char *shader_source_path, char *error, size_t error_size)
{
    (void)shader_source_path;
    if (error && error_size) error[0]=0; //;
    h3_gpu *gpu=(h3_gpu*)calloc(1,sizeof(*gpu));
    if(!gpu){ if(error&&error_size) snprintf(error,error_size,"oom"); return NULL; }
    int ndev=0; if(cudaGetDeviceCount(&ndev)!=cudaSuccess||ndev<=0){ if(error&&error_size) snprintf(error,error_size,"no CUDA devices"); free(gpu); return NULL; }
    gpu->device=0; if(cudaSetDevice(gpu->device)!=cudaSuccess){ if(error&&error_size) snprintf(error,error_size,"cudaSetDevice failed"); free(gpu); return NULL; }
    if(cudaStreamCreateWithFlags(&gpu->stream,cudaStreamNonBlocking)!=cudaSuccess){ if(error&&error_size) snprintf(error,error_size,"cudaStreamCreate failed"); free(gpu); return NULL; }
    if(cublasCreate(&gpu->cublas)!=CUBLAS_STATUS_SUCCESS){ if(error&&error_size) snprintf(error,error_size,"cublasCreate failed"); cudaStreamDestroy(gpu->stream); free(gpu); return NULL; }
    cublasSetStream(gpu->cublas,gpu->stream);
    if(cublasLtCreate(&gpu->cublaslt)!=CUBLAS_STATUS_SUCCESS){ if(error&&error_size) snprintf(error,error_size,"cublasLtCreate failed"); cublasDestroy(gpu->cublas); cudaStreamDestroy(gpu->stream); free(gpu); return NULL; }
    gpu->host_stage_bytes=64ull<<20; if(cudaMallocHost(&gpu->host_stage,gpu->host_stage_bytes)!=cudaSuccess){ gpu->host_stage=NULL; gpu->host_stage_bytes=0; }
    snprintf(gpu->profile_label,sizeof(gpu->profile_label),"cuda"); return gpu;
}

void h3_gpu_free(h3_gpu *gpu)
{
    if(!gpu) return; if(gpu->host_stage) cudaFreeHost(gpu->host_stage); if(gpu->cublaslt) cublasLtDestroy(gpu->cublaslt); if(gpu->cublas) cublasDestroy(gpu->cublas); if(gpu->stream) cudaStreamDestroy(gpu->stream); free(gpu);
}

int h3_gpu_is_m5(const h3_gpu *gpu)
{
    (void)gpu; return 0;
}

int h3_gpu_has_nax_mlp(const h3_gpu *gpu)
{
    (void)gpu; return 0;
}

int h3_gpu_has_int8_mlp(const h3_gpu *gpu)
{
    (void)gpu; return 0;
}

h3_gpu_tensor * h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements)
{
    return h3_tensor_alloc(gpu,elements,H3_GPU_F32);
}

h3_gpu_tensor * h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements)
{
    return h3_tensor_alloc(gpu,elements,H3_GPU_BF16);
}

h3_gpu_tensor * h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements)
{
    return h3_tensor_alloc(gpu,elements,H3_GPU_I8);
}

h3_gpu_tensor * h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values, size_t elements)
{
    h3_gpu_tensor *t=h3_tensor_alloc(gpu,elements,H3_GPU_F32); if(!t) return NULL; if(!h3_cuda_ok(gpu,cudaMemcpyAsync(t->device,values,elements*4u,cudaMemcpyHostToDevice,gpu->stream),"from_f32")||!h3_cuda_ok(gpu,cudaStreamSynchronize(gpu->stream),"from_f32 sync")){ h3_gpu_tensor_free(t); return NULL;} return t;
}

h3_gpu_tensor * h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values, size_t elements)
{
    h3_gpu_tensor *t=h3_tensor_alloc(gpu,elements,H3_GPU_BF16); if(!t) return NULL; if(!h3_cuda_ok(gpu,cudaMemcpyAsync(t->device,values,elements*2u,cudaMemcpyHostToDevice,gpu->stream),"from_bf16")||!h3_cuda_ok(gpu,cudaStreamSynchronize(gpu->stream),"from_bf16 sync")){ h3_gpu_tensor_free(t); return NULL;} return t;
}

h3_gpu_tensor * h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values, size_t elements)
{
    h3_gpu_tensor *t=h3_tensor_alloc(gpu,elements,H3_GPU_U32); if(!t) return NULL; if(!h3_cuda_ok(gpu,cudaMemcpyAsync(t->device,values,elements*4u,cudaMemcpyHostToDevice,gpu->stream),"from_u32")||!h3_cuda_ok(gpu,cudaStreamSynchronize(gpu->stream),"from_u32 sync")){ h3_gpu_tensor_free(t); return NULL;} return t;
}

h3_gpu_tensor * h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements)
{
    h3_gpu_tensor *t=h3_tensor_alloc(gpu,elements,H3_GPU_BF16); if(!t) return NULL; char err[256]; if(h3_gpu_tensor_read_file_bf16(t,path,file_offset,elements,err,sizeof(err))!=0){ h3_gpu_tensor_free(t); return NULL;} return t;
}

h3_gpu_tensor * h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path, uint64_t file_offset, size_t elements)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_tensor_load_f32");
    return NULL;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path, uint64_t file_offset, size_t elements, char *error, size_t error_size)
{
    if(!tensor||!path||tensor->dtype!=H3_GPU_BF16||elements>tensor->elements) return -1;
    if(error&&error_size) error[0]=0; //;
    h3_gpu *g=tensor->gpu; FILE *f=fopen(path,"rb");
    if(!f){ if(error&&error_size) snprintf(error,error_size,"open %s: %s",path,strerror(errno)); if(g) h3_set_error(g,"open %s: %s",path,strerror(errno)); return -1; }
    if(fseeko(f,(off_t)file_offset,SEEK_SET)!=0){ fclose(f); return -1; }
    size_t bytes=(size_t)elements*2u, off=0;
    while(off<bytes){ size_t chunk=bytes-off; if(g&&g->host_stage_bytes&&chunk>g->host_stage_bytes) chunk=g->host_stage_bytes; void *stage=g?g->host_stage:NULL; int tmp=0; if(!stage){ stage=malloc(chunk); tmp=1; if(!stage){ fclose(f); return -1; } } if(fread(stage,1,chunk,f)!=chunk){ if(tmp) free(stage); fclose(f); return -1; } if(!h3_cuda_ok(g,cudaMemcpyAsync((char*)tensor->device+off,stage,chunk,cudaMemcpyHostToDevice,g->stream),"read_file_bf16")){ if(tmp) free(stage); fclose(f); return -1; } if(tmp) free(stage); off+=chunk; if(g) g->stats.blit_copies++; }
    fclose(f); return h3_cuda_ok(g,cudaStreamSynchronize(g->stream),"read_file_bf16 sync")?0:-1;
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path, uint64_t file_offset, size_t elements, char *error, size_t error_size)
{
    return h3_gpu_tensor_read_file_bf16(tensor,path,file_offset,elements,error,error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor)
{
    if(!tensor) return; if(tensor->owned && tensor->device){ h3_stats_sub(tensor->gpu,tensor->bytes); cudaFree(tensor->device);} free(tensor);
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor)
{
    return tensor?tensor->elements:0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor)
{
    return tensor?tensor->dtype:H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values, size_t elements)
{
    if(!tensor||!values||tensor->dtype!=H3_GPU_F32||elements>tensor->elements) return -1; h3_gpu *g=tensor->gpu; if(!h3_cuda_ok(g,cudaMemcpyAsync(values,tensor->device,elements*4u,cudaMemcpyDeviceToHost,g->stream),"read_f32")) return -1; return h3_cuda_ok(g,cudaStreamSynchronize(g->stream),"read_f32 sync")?0:-1;
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor, size_t source_offset, float *values, size_t elements)
{
    return -1;
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values, size_t elements)
{
    if(!tensor||!values||tensor->dtype!=H3_GPU_BF16||elements>tensor->elements) return -1; h3_gpu *g=tensor->gpu; if(!h3_cuda_ok(g,cudaMemcpyAsync(values,tensor->device,elements*2u,cudaMemcpyDeviceToHost,g->stream),"read_bf16")) return -1; return h3_cuda_ok(g,cudaStreamSynchronize(g->stream),"read_bf16 sync")?0:-1;
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values, size_t elements)
{
    if(!tensor||!values||tensor->dtype!=H3_GPU_F32||elements>tensor->elements) return -1; h3_gpu *g=tensor->gpu; if(!h3_cuda_ok(g,cudaMemcpyAsync(tensor->device,values,elements*4u,cudaMemcpyHostToDevice,g->stream),"write_f32")) return -1; return h3_cuda_ok(g,cudaStreamSynchronize(g->stream),"write_f32 sync")?0:-1;
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor, size_t destination_offset, const float *values, size_t elements)
{
    return -1;
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values, size_t elements)
{
    if(!tensor||!values||tensor->dtype!=H3_GPU_BF16||elements>tensor->elements) return -1; h3_gpu *g=tensor->gpu; if(!h3_cuda_ok(g,cudaMemcpyAsync(tensor->device,values,elements*2u,cudaMemcpyHostToDevice,g->stream),"write_bf16")) return -1; return h3_cuda_ok(g,cudaStreamSynchronize(g->stream),"write_bf16 sync")?0:-1;
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor, size_t destination_offset, const uint16_t *values, size_t elements)
{
    return -1;
}

int h3_gpu_begin(h3_gpu *gpu)
{
    if(!gpu) return -1; gpu->has_error=0; gpu->error[0]=0; //; return 0;
}

int h3_gpu_continue(h3_gpu *gpu)
{
    if(!gpu) return -1; return gpu->has_error?-1:0;
}

int h3_gpu_submit(h3_gpu *gpu)
{
    if(!gpu) return -1; if(gpu->has_error) return -1; gpu->stats.submissions++; return h3_cuda_ok(gpu,cudaStreamSynchronize(gpu->stream),"submit")?0:-1;
}

const char * h3_gpu_error(const h3_gpu *gpu)
{
    return gpu?gpu->error:"null gpu";
}

int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats)
{
    if(!gpu||!stats) return -1; *stats=gpu->stats; return 0;
}

void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label)
{
    if(!gpu) return; snprintf(gpu->profile_label,sizeof(gpu->profile_label),"%s", label?label:"cuda");
}

void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase)
{
    (void)gpu; (void)phase;
}

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    if (!gpu || !output || !input || !weight) return -1;
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasSgemm(gpu->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)output_dim, (int)rows, (int)input_dim, &alpha,
        (const float *)weight->device, (int)output_dim,
        (const float *)input->device, (int)input_dim,
        &beta, (float *)output->device, (int)output_dim);
    if (!h3_cublas_ok(gpu, st, "linear_f32")) return -1;
    if (bias) {
        h3_k_bias_add_f32<<<h3_blocks((size_t)rows * output_dim), 256, 0, gpu->stream>>>(
            (float *)output->device, (const float *)bias->device, rows, output_dim);
        if (!h3_cuda_ok(gpu, cudaGetLastError(), "linear_f32 bias")) return -1;
    }
    gpu->stats.mps_linear_dispatches++;
    return 0;
}


int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_patch_linear_bf16");
    return -1;
}

int h3_gpu_patch_linear_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output, size_t output_offset, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_patch_linear_bf16_offset");
    return -1;
}

int h3_gpu_patch_linear_bf16_map(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, const h3_gpu_tensor *row_map, uint32_t output_rows, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_patch_linear_bf16_map");
    return -1;
}

int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if(!gpu||!output||!input) return -1; h3_k_silu_f32<<<h3_blocks(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)input->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"silu_f32")?0:-1;
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if(!gpu||!output||!input) return -1; h3_k_cast_f32_to_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const float*)input->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"cast_f32_to_bf16")?0:-1;
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if(!gpu||!output||!input) return -1; h3_k_cast_bf16_to_f32<<<h3_blocks(elements),256,0,gpu->stream>>>((float*)output->device,(const __nv_bfloat16*)input->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"cast_bf16_to_f32")?0:-1;
}

int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination, size_t destination_offset, const h3_gpu_tensor *source, size_t source_offset, size_t elements)
{
    if(!gpu||!destination||!source) return -1; size_t bytes=(size_t)elements*2u; const char *src=(const char*)source->device+(size_t)source_offset*2u; char *dst=(char*)destination->device+(size_t)destination_offset*2u; if(!h3_cuda_ok(gpu,cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,gpu->stream),"h3_gpu_copy_bf16")) return -1; gpu->stats.blit_copies++; gpu->stats.direct_dispatches++; return 0;
}

int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination, size_t destination_offset, const h3_gpu_tensor *source, size_t source_offset, size_t elements)
{
    if(!gpu||!destination||!source) return -1; size_t bytes=(size_t)elements*4u; const char *src=(const char*)source->device+(size_t)source_offset*4u; char *dst=(char*)destination->device+(size_t)destination_offset*4u; if(!h3_cuda_ok(gpu,cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,gpu->stream),"h3_gpu_copy_f32")) return -1; gpu->stats.blit_copies++; gpu->stats.direct_dispatches++; return 0;
}

int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, uint32_t rows, uint32_t width, float epsilon)
{
    if(!gpu||!output||!input) return -1; h3_k_rms_norm_f32<<<rows,256,0,gpu->stream>>>((float*)output->device,(const float*)input->device,weight?(const float*)weight->device:nullptr,rows,width,epsilon); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"rms_norm_f32")?0:-1;
}

int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_adaln_f32");
    return -1;
}

int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_gate_f32");
    return -1;
}

int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_qkv_rope_f32");
    return -1;
}

int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_sdpa_f32");
    return -1;
}

int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *fused, uint32_t rows, uint32_t width)
{
    if (!gpu || !output || !fused) return -1;
    size_t n = (size_t)rows * (size_t)width;
    h3_k_swiglu_fused_f32<<<h3_blocks(n), 256, 0, gpu->stream>>>((float *)output->device, (const float *)fused->device, rows, width);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "swiglu_f32") ? 0 : -1;
}


int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *scale, uint32_t rows, uint32_t width)
{
    if (!gpu || !output || !residual || !branch) return -1;
    size_t n = (size_t)rows * (size_t)width;
    h3_k_scale_add_tensor_f32<<<h3_blocks(n), 256, 0, gpu->stream>>>(
        (float *)output->device, (const float *)residual->device, (const float *)branch->device,
        scale ? (const float *)scale->device : nullptr, rows, width);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "scale_add_f32") ? 0 : -1;
}


int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_layer_norm_f32");
    return -1;
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_video_qkv_rope_f32");
    return -1;
}

int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t padding, uint32_t dilation)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_conv1d_f32");
    return -1;
}

int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding, uint32_t dilation)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_conv1d_stride_f32");
    return -1;
}

int h3_gpu_conv_transpose1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t length, uint32_t input_channels, uint32_t output_channels, uint32_t kernel, uint32_t stride, uint32_t padding)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_conv_transpose1d_f32");
    return -1;
}

int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *vector, const h3_gpu_tensor *magnitude, uint32_t outer, uint32_t inner)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_weight_norm_f32");
    return -1;
}

int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *left, const h3_gpu_tensor *right, float left_scale, float right_scale, uint32_t elements)
{
    if(!gpu||!output||!left||!right) return -1; h3_k_add_scaled_f32<<<h3_blocks(elements),256,0,gpu->stream>>>((float*)output->device,(const float*)left->device,(const float*)right->device,left_scale,right_scale,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"add_scaled_f32")?0:-1;
}

int h3_gpu_alias_free_snake_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha_log, const h3_gpu_tensor *beta_log, const h3_gpu_tensor *upsample_filter, const h3_gpu_tensor *downsample_filter, uint32_t batch, uint32_t length, uint32_t channels)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_alias_free_snake_f32");
    return -1;
}

int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *alpha, uint32_t batch, uint32_t length, uint32_t channels)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_snake1d_f32");
    return -1;
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_bias, const h3_gpu_tensor *k_bias, const h3_gpu_tensor *v_bias, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_audio_qkv_split_f32");
    return -1;
}

int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t batch, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_sdpa_causal_f32");
    return -1;
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *attended, uint32_t batch, uint32_t length, uint32_t heads, uint32_t head_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_audio_attention_pool_f32");
    return -1;
}

int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *linear, uint32_t elements)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_geglu_f32");
    return -1;
}

int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements, float minimum, float maximum)
{
    if(!gpu||!output||!input) return -1; h3_k_clip_f32<<<h3_blocks(maximum),256,0,gpu->stream>>>((float*)output->device,(const float*)input->device,elements,minimum,maximum); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"clip_f32")?0:-1;
}

int h3_gpu_vae_encoder_pad_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t depth_front, uint32_t height_before, uint32_t height_after, uint32_t width_before, uint32_t width_after)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_vae_encoder_pad_f32");
    return -1;
}

int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t input_channels, uint32_t output_channels, uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width, uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_conv3d_f32");
    return -1;
}

int h3_gpu_vae_encoder_group_norm_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width, uint32_t channels, uint32_t groups, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_vae_encoder_group_norm_silu_f32");
    return -1;
}

int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t input_dim, uint32_t output_dim)
{
    if (!gpu || !output || !input || !weight) return -1;
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmEx(gpu->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        (int)output_dim, (int)rows, (int)input_dim, &alpha,
        weight->device, CUDA_R_16BF, (int)output_dim,
        input->device, CUDA_R_16BF, (int)input_dim,
        &beta, output->device, CUDA_R_16BF, (int)output_dim,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!h3_cublas_ok(gpu, st, "linear_bf16")) return -1;
    if (bias) {
        h3_k_bias_add_bf16<<<h3_blocks((size_t)rows * output_dim), 256, 0, gpu->stream>>>(
            (__nv_bfloat16 *)output->device, (const __nv_bfloat16 *)bias->device, rows, output_dim);
        if (!h3_cuda_ok(gpu, cudaGetLastError(), "linear_bf16 bias")) return -1;
    }
    gpu->stats.mps_linear_dispatches++;
    return 0;
}


int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_mlp_bf16");
    return -1;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc2_weight, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_mlp_nax_bf16");
    return -1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *scales, const h3_gpu_tensor *input, uint32_t rows, uint32_t columns)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_quantize_weight_int8");
    return -1;
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t input_dim, uint32_t output_dim, int use_slower_uncached_int8_scales)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_linear_int8_bf16");
    return -1;
}

int h3_gpu_linear_int8_head_major_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, uint32_t rows, uint32_t heads, uint32_t head_dim, uint32_t output_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_linear_int8_head_major_bf16");
    return -1;
}

int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *activated, h3_gpu_tensor *quantized_activation, h3_gpu_tensor *activation_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *fc1_weight, const h3_gpu_tensor *fc1_scales, const h3_gpu_tensor *fc2_weight, const h3_gpu_tensor *fc2_scales, const h3_gpu_tensor *fc1_bf16, const h3_gpu_tensor *fc2_bf16, uint32_t rows, uint32_t input_dim, uint32_t hidden_dim, uint32_t output_dim, int use_slower_grouped_quantizer, int use_slower_dynamic_fc1_k, int use_int8_row_fc2, int input_is_quantized)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_mlp_int8_bf16");
    return -1;
}

int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements)
{
    if(!gpu||!output||!input) return -1; h3_k_silu_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"silu_bf16")?0:-1;
}

int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, uint32_t rows, uint32_t width, float epsilon)
{
    if(!gpu||!output||!input) return -1; h3_k_rms_norm_bf16<<<rows,256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device,weight?(const __nv_bfloat16*)weight->device:nullptr,rows,width,epsilon); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"rms_norm_bf16")?0:-1;
}

int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_layer_norm_bf16");
    return -1;
}

int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, uint32_t elements, int approximate)
{
    if(!gpu||!output||!input) return -1; (void)approximate; h3_k_gelu_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)input->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"gelu_bf16")?0:-1;
}

int h3_gpu_vision_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_vision_qkv_rope_bf16");
    return -1;
}

int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_adaln_bf16");
    return -1;
}

int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_adaln_bf16_offset");
    return -1;
}

int h3_gpu_adaln_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output, h3_gpu_tensor *inverse, const h3_gpu_tensor *input, size_t input_offset, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias, uint32_t rows, uint32_t width, uint32_t output_dim, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_adaln_linear_bf16");
    return -1;
}

int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_gate_bf16");
    return -1;
}

int h3_gpu_gate_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *output, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_gate_adaln_bf16");
    return -1;
}

int h3_gpu_gate_adaln_quantize_int8(h3_gpu *gpu, h3_gpu_tensor *gated_residual, h3_gpu_tensor *quantized_output, h3_gpu_tensor *quantized_scales, const h3_gpu_tensor *residual, const h3_gpu_tensor *branch, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *gate_modulation, const h3_gpu_tensor *norm_modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t padded_rows, uint32_t width, uint32_t slots, uint32_t gate_slot, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_gate_adaln_quantize_int8");
    return -1;
}

int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_qkv_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, const h3_gpu_tensor *qkv, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_grouped_qkv_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, h3_gpu_tensor *qkv, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_grouped_qkv_linear_rope_bf16");
    return -1;
}

int h3_gpu_grouped_qkv_linear_rope_int8(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, h3_gpu_tensor *value, h3_gpu_tensor *quantized_input, h3_gpu_tensor *input_scales, const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *weight_scales, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t rows, uint32_t input_dim, uint32_t heads, uint32_t head_dim, uint32_t rope_half, float epsilon, int input_is_quantized, int use_slower_unfused_qkv_rope, int use_slower_scalar_qkv_rms, int use_slower_uncached_int8_scales)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_grouped_qkv_linear_rope_int8");
    return -1;
}

int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_sdpa_bf16");
    return -1;
}

int h3_gpu_sdpa_bf16_head_major_output(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t heads, uint32_t head_dim, float scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_sdpa_bf16_head_major_output");
    return -1;
}

int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *fused, uint32_t rows, uint32_t width)
{
    if (!gpu || !output || !fused) return -1;
    size_t n = (size_t)rows * (size_t)width;
    h3_k_swiglu_fused_bf16<<<h3_blocks(n), 256, 0, gpu->stream>>>((__nv_bfloat16 *)output->device, (const __nv_bfloat16 *)fused->device, rows, width);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "swiglu_bf16") ? 0 : -1;
}


int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *weight, const h3_gpu_tensor *token_ids, uint32_t tokens, uint32_t vocab_size, uint32_t width)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_embedding_bf16");
    return -1;
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query_output, h3_gpu_tensor *key_output, const h3_gpu_tensor *query_input, const h3_gpu_tensor *key_input, const h3_gpu_tensor *q_norm, const h3_gpu_tensor *k_norm, const h3_gpu_tensor *rope_cos, const h3_gpu_tensor *rope_sin, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_text_qk_rope_bf16");
    return -1;
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor, const h3_gpu_tensor *weight, uint32_t sequence, uint32_t heads, uint32_t head_dim, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_head_rms_norm_bf16");
    return -1;
}

int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query, h3_gpu_tensor *key, const h3_gpu_tensor *rope_cos_f32, const h3_gpu_tensor *rope_sin_f32, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_rope_text_bf16");
    return -1;
}

int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *query, const h3_gpu_tensor *key, const h3_gpu_tensor *value, uint32_t sequence, uint32_t query_heads, uint32_t kv_heads, uint32_t head_dim, float scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_gqa_causal_bf16");
    return -1;
}

int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *left, const h3_gpu_tensor *right, uint32_t elements)
{
    if(!gpu||!output||!left||!right) return -1; h3_k_add_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)left->device,(const __nv_bfloat16*)right->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"add_bf16")?0:-1;
}

int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *left, const h3_gpu_tensor *right, uint32_t elements)
{
    if(!gpu||!output||!left||!right) return -1; h3_k_sub_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)left->device,(const __nv_bfloat16*)right->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"sub_bf16")?0:-1;
}

int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original, size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs, uint32_t input_rows, uint32_t rows, uint32_t baseline_rows, uint32_t width)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_token_pool_bf16");
    return -1;
}

int h3_gpu_token_pool_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output, const h3_gpu_tensor *input, size_t input_offset, h3_gpu_tensor *original, size_t original_offset, h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *pairs, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t input_rows, uint32_t rows, uint32_t baseline_rows, uint32_t width, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_token_pool_adaln_bf16");
    return -1;
}

int h3_gpu_token_expand_delta_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *original, size_t original_offset, const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents, uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_token_expand_delta_bf16");
    return -1;
}

int h3_gpu_token_expand_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *residual, h3_gpu_tensor *output, const h3_gpu_tensor *original, size_t original_offset, const h3_gpu_tensor *reduced, const h3_gpu_tensor *baseline, size_t baseline_offset, const h3_gpu_tensor *baseline_indices, const h3_gpu_tensor *parents, const h3_gpu_tensor *norm_weight, const h3_gpu_tensor *modulation, const h3_gpu_tensor *row_map, uint32_t rows, uint32_t reduced_rows, uint32_t baseline_rows, uint32_t width, uint32_t exact_prefix_rows, float update_scale, uint32_t slots, uint32_t shift_slot, uint32_t scale_slot, float epsilon)
{
    if(gpu) h3_set_error(gpu,"CUDA stub not implemented: h3_gpu_token_expand_adaln_bf16");
    return -1;
}

int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample, size_t sample_offset, const h3_gpu_tensor *last, const h3_gpu_tensor *previous, uint32_t elements, float delta, float ratio)
{
    if (!gpu || !sample || !last) return -1;
    float *sample_ptr = (float *)sample->device + sample_offset;
    const __nv_bfloat16 *last_ptr = (const __nv_bfloat16 *)last->device;
    const __nv_bfloat16 *prev_ptr = previous ? (const __nv_bfloat16 *)previous->device : nullptr;
    h3_k_euler_bf16<<<h3_blocks(elements), 256, 0, gpu->stream>>>(sample_ptr, last_ptr, prev_ptr, elements, delta, ratio);
    gpu->stats.direct_dispatches++;
    return h3_cuda_ok(gpu, cudaGetLastError(), "euler_bf16") ? 0 : -1;
}


int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output, const h3_gpu_tensor *gate, const h3_gpu_tensor *up, uint32_t elements)
{
    if(!gpu||!output||!gate||!up) return -1; h3_k_swiglu_bf16<<<h3_blocks(elements),256,0,gpu->stream>>>((__nv_bfloat16*)output->device,(const __nv_bfloat16*)gate->device,(const __nv_bfloat16*)up->device,elements); gpu->stats.direct_dispatches++; return h3_cuda_ok(gpu,cudaGetLastError(),"h3_gpu_silu_mul_bf16")?0:-1;
}

