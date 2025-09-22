#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdint>

#include <map>
#include <queue>

int main() {
  int nodes_per_page = 7;
  int entry_point = 65610822;
  int node_size = 516;
  int data_size = 128;

  int page_size = 4096;

  const char* file = "/mnt/data/index/sift100M/disk_index_R96_L120_disk.index";

  FILE* f = fopen(file, "r");
  std::queue<int> q;
  std::map<int, int> mp;
  mp[entry_point] = 0;
  q.push(entry_point);
  while(!q.empty()) {
    int u = q.front(); q.pop();
    int d = mp[u];
    if (d > 9) break;
    int blk = u / nodes_per_page + 1;
    int offset = u % nodes_per_page;
    char buff[4096];
    fseek(f, 1l * blk * page_size, SEEK_SET);
    fread(buff, 1, 4096, f);
    char* start = buff + offset * node_size + data_size;
    int* edge = (int*) start;
    int len = edge[0];
    //printf("%d\n", len);
    edge++;
    for (int i = 0; i < len; i++) {
      int v = edge[i];
      if (mp.count(v)) continue;
      mp[v] = d + 1;
      q.push(v);
    }
  }
  printf("Done!\n");
  std::map<int, int> dist;
  int x;
  while(scanf("%d", &x) == 1) {
    dist[mp[x]]++;
  }
  for (auto [x, y]: dist) {
    printf("%d %d\n", x, y);
  }
}
