# 每日一推

- SVM 推导
- kalman 推导
- sort & deepsort 推导
- self-attention & transformer 推导
- 匈牙利算法，二分匹配
- 排序方法：冒泡、插入、快速、计数
 

## 传统图像
- 放射变换公式
- 形态学变换：开、闭、梯度、顶帽、底帽 
  ```python
    def close_demo_h(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))  
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary
    ```
  
  开运算：先腐蚀后膨胀  消除小物体。
  
  闭运算：先膨胀后腐蚀 排除小型黑洞。
    
  梯度：膨胀-腐蚀 保留物体边缘轮廓。
    
  顶帽：原图-开运算 背景提取。
    
  黑帽：闭运算-原图 轮廓提取。

## c++ 概念

## others
- Pruning Filters For Efficient ConvNets, ICLR2017
- distilling the Knowledge in a Neural Network

- docker, fastapi, git , locust 命令。
  
- torch,onnx,trt 的运行速度对比？
- c++的CV2图片怎么转为python格式的v2图片？
- ctc 原理与用法是？
- 

