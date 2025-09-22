#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <random>

int main(int argc, char** argv) {
  if (argc != 6) {
    printf("Usage: %s <sample num> <size> <input> <output data> <output map>\n", argv[0]);
    return -1;
  }
  int num_samp = atoi(argv[1]);
  int size = atoi(argv[2]);
  FILE* f = fopen(argv[3], "r");
  int num_data, data_len;
  fread((char*)&num_data, sizeof(int), 1, f);
  fread((char*)&data_len, sizeof(int), 1, f);
  std::mt19937 gen(114514);


  std::vector<int> vec;
  for (int i = 0; i < num_data; i++) vec.push_back(i);
  std::shuffle(vec.begin(), vec.end(), gen);
  vec.resize(num_samp);
  std::sort(vec.begin(), vec.end());
  FILE* of1 = fopen(argv[5], "w");
  for (auto x: vec) {
    fprintf(of1, "%d\n", x);
  }
  fclose(of1);
  FILE* of2 = fopen(argv[4], "w");
  fwrite((char*)&num_samp, sizeof(int), 1, of2);
  fwrite((char*)&data_len, sizeof(int), 1, of2);
  char* buff = new char[data_len * size];
  for (auto x: vec) {
    fseek(f, sizeof(int) * 2 + 1l * data_len * x * size, SEEK_SET);
    fread(buff, 1, data_len * size, f);
    fwrite(buff, 1, data_len * size, of2);
  }
  fclose(of2);
  fclose(f);
}
