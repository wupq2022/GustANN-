#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

constexpr int PAGE_SIZE = 4096;

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("%s: <input> <#shards>\n", argv[0]);
    return -1;
  }
  std::string fname(argv[1]);
  int shards = atoi(argv[2]);
  if (shards <= 0) {
    printf("Invalid #shard");
    return -1;
  }

  FILE* f0 = fopen(argv[1], "rb");
  fseek(f0, PAGE_SIZE, SEEK_SET);
  char buff[PAGE_SIZE];
  std::vector<FILE*> out_f;
  for (int i = 0; i < shards; i++) {
    out_f.push_back(fopen((fname + "." + std::to_string(shards) + "." + std::to_string(i + 1)).c_str(), "wb"));
  }

  int i = 0;
  while(true) {
    if (fread(buff, 1, PAGE_SIZE, f0) != PAGE_SIZE)
      break;
    
    fwrite(buff, 1, PAGE_SIZE, out_f[i]);
    i = (i + 1) % shards;
  }
  
}
