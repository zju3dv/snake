#include <float.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#define DIST(x1,y1,z1,x2,y2,z2) (((x1)-(x2))*((x1)-(x2))+((y1)-(y2))*((y1)-(y2))+((z1)-(z2))*((z1)-(z2)))
#define DIST2D(x1,y1,x2,y2) (((x1)-(x2))*((x1)-(x2))+((y1)-(y2))*((y1)-(y2)))

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

int infTwoExp(int val)
{
    int inf=1;
    while(val>inf) inf<<=1;
    return inf;
}

void getGPULayout(
        int dim0,int dim1,int dim2,
        int* bdim0,int* bdim1,int* bdim2,
        int* tdim0,int* tdim1,int* tdim2
)
{
    (*tdim2)=64;
    if(dim2<(*tdim2)) (*tdim2)=infTwoExp(dim2);
    (*bdim2)=dim2/(*tdim2);
    if(dim2%(*tdim2)>0) (*bdim2)++;

    (*tdim1)=1024/(*tdim2);
    if(dim1<(*tdim1)) (*tdim1)=infTwoExp(dim1);
    (*bdim1)=dim1/(*tdim1);
    if(dim1%(*tdim1)>0) (*bdim1)++;

    (*tdim0)=1024/((*tdim1)*(*tdim2));
    if(dim0<(*tdim0)) (*tdim0)=infTwoExp(dim0);
    (*bdim0)=dim0/(*tdim0);
    if(dim0%(*tdim0)>0) (*bdim0)++;
}
#endif
