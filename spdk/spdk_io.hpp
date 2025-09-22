#pragma once

#include <memory>
#include <vector>
#include <string>

class SpdkIO {
public:
  static std::shared_ptr<SpdkIO> create();
  virtual void init(const std::vector<std::string>&, int queue_cap, int thread_cnt, int ctx_cnt) = 0;
  virtual void push_queue(const std::vector<std::pair<int, void*>>&, int, int) = 0;
  virtual bool check_ready(int) = 0;
  virtual void print_stats(std::vector<double>) = 0;
  virtual void clear_stats() = 0;
  virtual ~SpdkIO() {}
};  
