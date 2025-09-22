import io_utils
import sys
import numpy as np

if (len(sys.argv) != 3):
    print(f"Usage: {sys.argv[0]} .txt .bin")
    exit(-1)

txt_file = sys.argv[1]
bin_file = sys.argv[2]
with open(txt_file) as f:
    data = list(map(int, filter(lambda x: len(x) > 0, map(lambda x: x.strip(), f.readlines()))))
    print(data[:100])
    data = np.array(data, dtype="int32")
    data = data.reshape((-1,1))
    print(data.shape)
    io_utils.ibin_write(bin_file, data)
