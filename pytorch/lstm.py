from unicodedata import bidirectional
import torch
import torch.nn as nn
import time
import os
import flush_cpp as fl

m = 576
torch.set_num_threads(1)

LSTM = torch.load('./weight/lstm.pt').eval()
x = torch.randn(1, m, 1024)
h0 = torch.randn(2, m, 512)
c0 = torch.randn(2, m, 512)

print("\n----lets run!----")
avg_time = 0

print("compute: lstm layer")
start = time.time() #####

x, (h, c) = LSTM(x, (h0, c0))

end = time.time()   #####
print("time: ", end-start)
