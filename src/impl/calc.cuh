#pragma once
namespace gustann {
  using namespace cuhnsw;
  #define WARP_SIZE 32

  template <class T>
  __inline__ __device__
  float square_sum(const float * a, T* b, const int num_dims) {
    __syncthreads();
    static __shared__ float shared[32];
    
    // figure out the warp/ position inside the warp
    int warp =  threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    float val = 0;
        
    for (int i = threadIdx.x; i < num_dims; i += blockDim.x) {
      float _val = a[i] - (float)(b[i]);
      val += _val * _val;
    }
    __syncthreads();
    val = warp_reduce_sum(val);
    
    // write out the partial reduction to shared memory if appropiate
    if (lane == 0) {
      shared[warp] = val;
    }
    __syncthreads();
    
    // if we we don't have multiple warps, we're done
    if (blockDim.x <= WARP_SIZE) {
      return shared[0];
    }
    
    // otherwise reduce again in the first warp
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane]: conversion(0.0f);
    if (warp == 0) {
      val = warp_reduce_sum(val);
      // broadcast back to shared memory
      if (threadIdx.x == 0) {
        shared[0] = val;
      }
    }
    __syncthreads();
    return shared[0];
  }

  template <class T>
  __inline__ __device__
  bool PushNodeToSearchPqBam(Neighbor* pq, int* size, const int max_size,
                             T* data, const int num_dims,
                             const float* src_vec, const int dstid) {

    if (CheckAlreadyExists(pq, *size, dstid)) return false;
    float dist = square_sum(src_vec, data, num_dims);
    bool ret = false;
    __syncthreads();
    if (*size < max_size) {
      PqPush(pq, size, dist, dstid, false);
      ret = true;
    } else if (gt(pq[0].distance, dist)) {
      PqPop(pq, size);
      PqPush(pq, size, dist, dstid, false);
      ret = true;
    }
    __syncthreads();
    return ret;
  }

  __inline__ __device__
  bool PushNodeToSearchPqBam(Neighbor* pq, int* size, const int max_size,
                             float dist,
                             const int dstid) {
    //float dist = square_sum(src_vec, data, num_dims);
    bool ret = false;
    __syncthreads();
    if (*size < max_size) {
      PqPush(pq, size, dist, dstid, false);
      ret = true;
    } else if (gt(pq[0].distance, dist)) {
      PqPop(pq, size);
      PqPush(pq, size, dist, dstid, false);
      ret = true;
    }
    __syncthreads();
    return ret;
  }

}