#pragma once
#include "types.hpp"
#include "common.hpp"
#include "pq_search.hpp"

namespace gustann {  
#ifdef _USE_BAM
//#define _IN_MEM

#ifdef _IN_MEM
  using DiskData = uint8_t;
#else
  using DiskData = array_d_t<uint8_t>;
#endif

  
  __global__ void search_disk_graph_kernel
  (float* qdata, const int num_qnodes,
   DiskData *data,
   //uint8_t* data,
   const int num_nodes, const int num_dims, const int max_m,
   const int ef_search, const int* entries, const int topk,
   int* nns, float* distances, int* found_cnt,
   int* visited_table, int* visited_list, 
   const int visited_table_size, const int visited_list_size, int64_t* acc_visited_cnt,
   Neighbor* neighbors, int node_per_sector, int node_len, int data_len, DataType data_type,
   PQSearchData* pq = nullptr);
#endif

  struct __align__(128) Data {
    int visited_cnt;
    int size;
  };

}
