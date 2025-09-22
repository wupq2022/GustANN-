#pragma once
#include <sstream>

#ifdef CHECK_CUDA
#undef CHECK_CUDA
#endif

#define CHECK_CUDA(code) { checkCuda((code), __FILE__, __LINE__); }
inline void checkCuda(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::stringstream err;
    err << "Cuda Error: " << cudaGetErrorString(code) << " (" << file << ":" << line << ")";
    throw std::runtime_error(err.str());
  }
}

template <class T>
inline void copy_to_dev(T* host_data, T* &dev_data, size_t len) {
  CHECK_CUDA(cudaMalloc(&dev_data, sizeof(T) * len));
  CHECK_CUDA(cudaMemcpy(dev_data, host_data, sizeof(T) * len, cudaMemcpyHostToDevice));
}


namespace gustann {
  struct BaMConfig {
    int num_ctrls = 1;
    int queue_depth = 1024;
    int num_queues = 1;
    int cuda_device = 0;
    int nvm_namespace = 1;
    int page_size = 4096;
    int num_page = 1024; // cached page
    bool use_simple_cache = false;
  };
  enum DataType {
    FLOAT,
    UINT8,
  };
}

