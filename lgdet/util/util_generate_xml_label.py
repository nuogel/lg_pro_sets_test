# encoding:utf-8
'''
根据一个给定的XML Schema，使用DOM树的形式从空白文件生成一个XML。
'''
from xml.dom.minidom import Document

doc = Document()  # 创建DOM文档对象
annotation = doc.createElement('annotation')  # 创建根元素
doc.appendChild(annotation)
# SIZE:
size = doc.createElement('size')
annotation.appendChild(size)
# size _children:
height = doc.createElement('height')
height.appendChild(doc.createTextNode('400'))
width = doc.createElement('width')
width.appendChild(doc.createTextNode('200'))
depth = doc.createElement('depth')
depth.appendChild(doc.createTextNode('3'))
size.appendChild(height)
size.appendChild(width)
size.appendChild(depth)


def add_object(_name, _difficult, _bbox):
    object = doc.createElement('object')
    annotation.appendChild(object)

    name = doc.createElement('name')
    name.appendChild(doc.createTextNode(_name))
    object.appendChild(name)

    difficult = doc.createElement('difficult')
    difficult.appendChild(doc.createTextNode(str(_difficult)))
    object.appendChild(difficult)

    bndbox = doc.createElement('bndbox')

    #  bboxes
    xmin = doc.createElement('xmin')
    xmin.appendChild(doc.createTextNode(str(_bbox[0])))
    bndbox.appendChild(xmin)

    ymin = doc.createElement('ymin')
    ymin.appendChild(doc.createTextNode(str(_bbox[1])))
    bndbox.appendChild(ymin)

    xmax = doc.createElement('xmax')
    xmax.appendChild(doc.createTextNode(str(_bbox[2])))
    bndbox.appendChild(xmax)

    ymax = doc.createElement('ymax')
    ymax.appendChild(doc.createTextNode(str(_bbox[3])))
    bndbox.appendChild(ymax)

    object.appendChild(bndbox)


add_object('person', 0, [10, 20, 30, 40])
add_object('CAR', 1, [110, 10, 310, 410])
add_object('CAR', 1, [110, 10, 310, 410])

#
########### 将DOM对象doc写入文件
f = open('annotation.xml', 'w', encoding='utf-8')
f.write(doc.toprettyxml(indent=''))
f.close()
