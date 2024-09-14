from PIL import Image, ImageDraw, ImageFont
import cv2


pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# pil_img.show()
# 生成画笔
draw = ImageDraw.Draw(pil_img)
# 第一个参数是字体文件的路径，第二个是字体大小
font = ImageFont.truetype('/home/dell/lg/rock_lg/utils/font/simhei.ttf', 60, encoding='utf-8')
draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])),outline='red',width=4)
draw.text((bbox[0], bbox[1]-61), '行人',fill='red', font=font)

img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=4)
img = cv2.putText(img, 's', (bbox[0] + 10, bbox[1] + 10), cv2.FONT_ITALIC, 0.5, color, 2)
