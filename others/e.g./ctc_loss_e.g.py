import torch
import torch.nn as nn

ctc_loss = nn.CTCLoss()
log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
targets = torch.randint(1, 20, (16, 30), dtype=torch.long) # 16 个序列， 最长30个长度， 每个20个向量。
input_lengths = torch.full((16,), 50, dtype=torch.long)
target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
loss.backward()
