#pragma once

// The four kernels every CSM translation unit was defining for itself.
//
// `csm_backbone_forward.cu`, `csm_depth_decoder_forward.cu` and
// `mimi_decoder_forward.cu` each carried a `k_matmul`; two of them also
// carried `k_rms`, `k_swiglu` and `k_argmax`. Nine definitions, four
// distinct bodies, byte-identical after comment stripping — a fingerprint
// check, not a reading. Anonymous-namespace kernels are invisible to the
// next file's author, which is the mechanism that produced them.
//
// This header is a WAY STATION, not the answer. The answer is calling the
// primitives that already exist (`gemm::act_x_wt_bf16`, `norm::rmsnorm_bf16`,
// `mlp::swiglu_bf16`, `sample::argmax_bf16`), and that swap changes the
// arithmetic — a naive scalar loop is not cuBLAS — so it needs the parity
// harness (`csm_backbone_parity.cu`) run against reference dumps. Collapsing
// nine identical bodies into four does NOT: same body, same launch config,
// same numbers. So it can land now and the swap can wait for the dumps.
//
// The kernels stay in an anonymous namespace: each TU still gets its own
// copy, which is exactly what it had. Only the source duplication goes.

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

__global__ void k_swiglu(const bf* gate,const bf* up,bf* o,long t){
    long i=blockIdx.x*(long)blockDim.x+threadIdx.x;if(i>=t)return;
    float g=F(gate[i]);o[i]=Bf((g/(1.f+__expf(-g)))*F(up[i]));
}

__global__ void k_argmax(const bf* logits,int V,int* out){
    int t=threadIdx.x;float bv=-1e30f;int bi=0;
    for(int v=t;v<V;v+=blockDim.x){float x=F(logits[v]);if(x>bv){bv=x;bi=v;}}
    __shared__ float sv[256];__shared__ int si[256];
    sv[t]=bv;si[t]=bi;__syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){if(t<s){if(sv[t+s]>sv[t]||(sv[t+s]==sv[t]&&si[t+s]<si[t])){sv[t]=sv[t+s];si[t]=si[t+s];}}__syncthreads();}
    if(t==0)*out=si[0];
}

}  // namespace
}  // namespace pie_cuda_driver::model
