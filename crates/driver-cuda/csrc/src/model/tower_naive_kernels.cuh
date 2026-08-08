#pragma once

// The naive kernels more than one TOWER defines identically.
//
// `k_matmul` and `k_rms` are byte-identical in `model/csm/` and
// `model/gemma4/` -- fingerprinted, not read. Every tower writes its kernels
// in an anonymous namespace, so no author can see the copy next door; that
// invisibility is the mechanism, and it produced the same scalar matmul three
// times and the same RMSNorm twice.
//
// This is a WAY STATION. The answer is to stop having them: `gemm::act_x_wt_bf16`
// and `norm::rmsnorm_bf16` already exist and are what these should call. That
// swap changes the arithmetic -- a naive scalar loop is not cuBLAS -- so it
// needs the tower parity harnesses (`gemma4_vision_full_parity_bf16`,
// `csm_backbone_parity`) run against reference dumps. Collapsing identical
// bodies does not, which is why this can land and that cannot.
//
// Names whose bodies DIFFER between towers stay where they are: `k_add`,
// `k_addpos`, `k_attn`, `k_f32_to_bf16`, `k_gelu`, `k_layernorm`, `k_rope`.
// Each needs reading before anyone claims two of them are the same op.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace pie_cuda_driver::model {
namespace {

using bf = __nv_bfloat16;
__device__ __forceinline__ float F(bf x){return __bfloat162float(x);}
__device__ __forceinline__ bf   Bf(float x){return __float2bfloat16(x);}

__global__ void k_matmul(const bf* x,const bf* W,bf* y,int N,int K,int O){
    int n=blockIdx.y*blockDim.y+threadIdx.y,o=blockIdx.x*blockDim.x+threadIdx.x;
    if(n>=N||o>=O)return;
    const bf* xr=x+(long)n*K;const bf* wr=W+(long)o*K;
    float a=0;for(int k=0;k<K;k++)a+=F(xr[k])*F(wr[k]);
    y[(long)n*O+o]=Bf(a);
}

__global__ void k_rms(const bf* x,const bf* w,bf* o,int R,int D,float eps){
    int r=blockIdx.x;if(r>=R)return;const bf* xr=x+(long)r*D;bf* orow=o+(long)r*D;
    float loc=0;for(int d=threadIdx.x;d<D;d+=blockDim.x){float v=F(xr[d]);loc+=v*v;}
    for(int s=warpSize/2;s>0;s>>=1)loc+=__shfl_down_sync(0xffffffff,loc,s);
    __shared__ float warp[32],ss;if((threadIdx.x&31)==0)warp[threadIdx.x>>5]=loc;__syncthreads();
    if(threadIdx.x==0){float t=0;int nw=(blockDim.x+31)/32;for(int i=0;i<nw;i++)t+=warp[i];ss=rsqrtf(t/D+eps);}__syncthreads();
    float inv=ss;for(int d=threadIdx.x;d<D;d+=blockDim.x)orow[d]=Bf(F(xr[d])*inv*(w?F(w[d]):1.f));
}

}  // namespace
}  // namespace pie_cuda_driver::model
