import lmdb

image_path = '/media/lg/SSD_WorkSpace/LG/GitHub/lg_pro_sets/saved/efficientdet_kitti_512x768.png'
label = 'cat'
map_size = 1e9
env = lmdb.open('lmdb_dir', map_size=map_size)
cache = {}  # 存储键值对

with open(image_path, 'rb') as f:
    # 读取图像文件的二进制格式数据
    image_bin = f.read()

# 用两个键值对表示一个数据样本
cache['image_000'] = image_bin
cache['label_000'] = label

with env.begin(write=True) as txn:
    for k, v in cache.items():
        if isinstance(v, bytes):
            # 图片类型为bytes
            txn.put(k.encode(), v)
        else:
            # 标签类型为str, 转为bytes
            txn.put(k.encode(), v.encode())  # 编码

env.close()