OCR = DET + REC

DET:文本检测算法有DB、EAST、SAST等等

REC:文本识别算法有CRNN、RARE、StarNet、Rosetta、SRN等算法。

### 1. OCR 检测模型基本概念

文本检测就是要定位图像中的文字区域，然后通常以边界框的形式将单词或文本行标记出来。传统的文字检测算法多是通过手工提取特征的方式，特点是速度快，简单场景效果好，但是面对自然场景，效果会大打折扣。当前多是采用深度学习方法来做。

基于深度学习的文本检测算法可以大致分为以下几类：
1. 基于目标检测的方法；一般是预测得到文本框后，通过NMS筛选得到最终文本框，多是四点文本框，对弯曲文本场景效果不理想。典型算法为EAST、Text Box等方法。
2. 基于分割的方法；将文本行当成分割目标，然后通过分割结果构建外接文本框，可以处理弯曲文本，对于文本交叉场景问题效果不理想。典型算法为DB、PSENet等方法。
3. 混合目标检测和分割的方法；

---
### 2. OCR 识别模型基本概念

OCR识别算法的输入数据一般是文本行，背景信息不多，文字占据主要部分，识别算法目前可以分为两类算法：
1. 基于CTC的方法；即识别算法的文字预测模块是基于CTC的，常用的算法组合为CNN+RNN+CTC。目前也有一些算法尝试在网络中加入transformer模块等等。
2. 基于Attention的方法；即识别算法的文字预测模块是基于Attention的，常用算法组合是CNN+RNN+Attention。

---

###3.tricks

#### 3.1 CTCLOSS
ctc_loss = nn.CTCLoss(blank=0, reduction='mean') #blank:空白标签所在位子，默认第0个；
loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

- log_probs：shape为(T, N, C)的模型输出张量，其中，T表示CTCLoss的输入长度也即输出序列长度，N表示训练的batch size长度，C则表示包含有空白标签的所有要预测的字符集总长度，log_probs一般需要经过torch.nn.functional.log_softmax处理后再送入到CTCLoss中；
- targets：shape为(N, S) 或(sum(target_lengths))的张量，其中第一种类型，N表示训练的batch size长度，S则为标签长度，第二种类型，则为所有标签长度之和，但是需要注意的是targets不能包含有空白标签；
- input_lengths：shape为(N)的张量或元组，但每一个元素的长度必须等于T即输出序列长度，一般来说模型输出序列固定后则该张量或元组的元素值均相同；
- target_lengths：shape为(N)的张量或元组，其每一个元素指示每个训练输入序列的标签长度，但标签长度是可以变化的；

e.g: [官方所给的例程如下](../others/e.g./ctc_loss_e.g.py)，但在实际应用中需要将log_probs的detach()去掉，否则无法反向传播进行训练
```python
    >>> ctc_loss = nn.CTCLoss()
    >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
    >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
    >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
    >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
    >>> loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    >>> loss.backward()
```