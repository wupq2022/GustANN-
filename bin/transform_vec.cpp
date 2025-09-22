#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <iostream>
#include <fstream>

struct MetaData {
  uint32_t nr;
  uint32_t nc;
  uint64_t nnodes;
  uint64_t ndims;
  uint64_t medoid_id_on_file;
  uint64_t max_node_len;
  uint64_t nnodes_per_sector;
  uint64_t num_frozen_points;
  uint64_t file_frozen_id;
  uint64_t reorder_data_exists;
} __attribute__((packed));

template<class T>
static void read_bin(std::string filename, int& npts, int& ndim, size_t offset, T* &data) {
  std::ifstream ifile(filename);
  if (!ifile.is_open()) {
    printf("Cannot open file: %s\n", filename.c_str());
    return;
  }
  ifile.seekg(offset, std::ios::beg);
  ifile.read((char*)&npts, sizeof(int));
  ifile.read((char*)&ndim, sizeof(int));
  size_t len = (size_t)npts * ndim;
  data = new T [len];
  ifile.read((char*)data, sizeof(T) * len);
  printf("Read %d x %d data from file: %s\n", npts, ndim, filename.c_str());
}

template<class T>
static size_t write_bin(FILE* ofile, int npts, int ndim, size_t offset, T* data) {
  assert(fseek(ofile, offset, SEEK_SET) == 0);
  printf("%ld\n", ftell(ofile));
  assert(fwrite(&npts, sizeof(int), 1, ofile) == 1);
  assert(fwrite(&ndim, sizeof(int), 1, ofile) == 1);
  size_t len = (size_t)npts * ndim;
  assert(fwrite(data, sizeof(T), len, ofile) == len);
  printf("Write %d x %d data @ %lu\n", npts, ndim, offset);
  printf("%lu\n", len * sizeof(T));
  size_t ret = ftell(ofile);
  return ret;
}

using data_type = float;

int main(int argc, char** argv) {
  if (argc != 8) {
    printf("Usage: %s <Input Index> <Input PQ> <Input Nav Data> <Replicate Factor> <Output Index> <Output PQ> <Input Nav Data> <Output Nav Data>\n", argv[0]);
    return -1;
  }

  FILE* inpf = fopen(argv[1], "r");
  MetaData data;
  assert(fread(&data, sizeof(MetaData), 1, inpf) == 1);

  printf("%lu %lu %lu %lu\n", data.nnodes, data.ndims, data.max_node_len, data.nnodes_per_sector);

  int old_dims = data.ndims;
  int old_node_len = data.max_node_len;
  int old_nnodes_per_sector = data.nnodes_per_sector;
  

  constexpr int BUFF_SIZE = 4096;
  constexpr int data_size = sizeof(data_type);
  char inbuff[BUFF_SIZE];
  char outbuff[BUFF_SIZE];

  int data_len = data.ndims * data_size;
  int edge_len = data.max_node_len - data.ndims * data_size;
  int rep = atoi(argv[3]);
  
  assert(rep > 1);
  data.ndims *= rep;
  data.max_node_len = data.ndims * data_size + edge_len;
  assert(data.max_node_len <= BUFF_SIZE);
  data.nnodes_per_sector = BUFF_SIZE / data.max_node_len;

  FILE* oupf = fopen(argv[4], "wb");
  
  assert(fwrite(&data, sizeof(MetaData), 1, oupf) == 1);

  printf("%d %d %d\n", data_len, data.max_node_len, edge_len);
  for (int i = 0; i < data.nnodes; i++) {
    int64_t old_page = i / old_nnodes_per_sector + 1;
    int64_t old_offset = i % old_nnodes_per_sector * old_node_len;
    fseek(inpf, old_page * BUFF_SIZE + old_offset, SEEK_SET);
    assert(fread(inbuff, 1, old_node_len, inpf) == old_node_len);
    data_type* out_data = (data_type*) outbuff;
    data_type* in_data = (data_type*) inbuff;


    for (int k = 0; k < old_dims; k++) {
      for (int j = 0; j < rep; j++) {
        out_data[k * rep + j] = in_data[k];
      }
    }
    for (int k = 0; k < edge_len; k++) {
      outbuff[rep * data_len + k] = inbuff[data_len + k];
    }
    int64_t new_page = i / data.nnodes_per_sector + 1;
    int64_t new_offset = i % data.nnodes_per_sector * data.max_node_len;
    fseek(oupf, new_page * BUFF_SIZE + new_offset, SEEK_SET);
    assert(fwrite(outbuff, 1, data.max_node_len, oupf) == data.max_node_len);

    if (i % 1000000 == 0) {
      printf("Processed %d\n", i);
    }
  }
  int64_t last_page = (data.nnodes + data.nnodes_per_sector - 1) / data.nnodes_per_sector + 1 + 1;
  printf("%ld\n", last_page);
  fseek(oupf, last_page * BUFF_SIZE, SEEK_SET);
  fwrite(outbuff, 1, 1, oupf);

  fclose(inpf);
  fclose(oupf);

  const char* in_pq = argv[2];
  FILE* out_pq = fopen(argv[5], "w+");
  size_t* basic_offsets;
  int nr, nc;
  
  read_bin(in_pq, nr, nc, 0, basic_offsets);
  printf("%lu %lu %lu %lu\n", basic_offsets[0], basic_offsets[1], basic_offsets[2], basic_offsets[3]);
  assert(nr == 4 && nc == 1);
  size_t new_offsets[4];
  new_offsets[0] = basic_offsets[0];

  
  int dim, num_pivot;
  float* pivots;
  
  read_bin(in_pq, num_pivot, dim, basic_offsets[0], pivots);
  assert(num_pivot == 256);
  float* new_pivots = new float[rep * num_pivot * dim];
  for (int i = 0; i < num_pivot; i++) {
    for (int j = 0; j < dim; j++) {
      for (int k = 0; k < rep; k++) {
        new_pivots[(i * dim + j) * rep + k] = pivots[i * dim + j];
      }
    }
  }
  new_offsets[1] = write_bin(out_pq, num_pivot, dim * rep, new_offsets[0], new_pivots);
  printf("%lu\n", new_offsets[1]);

  float* centroid;
  read_bin(in_pq, nr, nc, basic_offsets[1], centroid);
  assert(nr == dim && nc == 1);
  float* new_centroid = new float[dim * rep];
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < rep; j++) {
      new_centroid[i * rep + j] = centroid[i];
    }
  }
  new_offsets[2] = write_bin(out_pq, dim * rep, 1, new_offsets[1], new_centroid);

  int* chunk_offsets;
  read_bin(in_pq, nr, nc, basic_offsets[2], chunk_offsets);
  assert(nc == 1);
  for (int i = 0; i < nr; i ++) {
    chunk_offsets[i] *= rep;
  }
  new_offsets[3] = write_bin(out_pq, nr, nc, new_offsets[2], chunk_offsets);
  write_bin(out_pq, 4, 1, 0, new_offsets);


  data_type* navdata;
  read_bin(argv[6], nr, nc, 0, navdata);
  data_type* new_navdata = new data_type [(int64_t) nr * nc * rep];
  for (int64_t i = 0; i < nr; i++) {
    for (int64_t j = 0; j < nc; j++) {
      for (int64_t k = 0; k < rep; k++) {
        new_navdata[(i * nc + j) * rep + k] = navdata[i * nc + j];
      }
    }
  }
  write_bin(fopen(argv[7], "w"), nr, nc * rep, 0, new_navdata);
  
  
}
