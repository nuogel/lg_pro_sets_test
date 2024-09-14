import os
import cv2


def images2video(images_dir, video_save_path, fps=10):
    # fourcc = cv2.VideoWriter_fourcc('H', '2', '6', '4')
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    videoWriter = cv2.VideoWriter(video_save_path, fourcc, fps, (1920, 1080))
    files = os.listdir(images_dir)
    count = 0
    for file in files:
        count += 1
        print(count)
        img_path = os.path.join(images_dir, file)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (1920, 1080))
        videoWriter.write(image)
    videoWriter.release()

    print("finished !")


if __name__ == '__main__':
    # img_path='/media/dell/data/garbage/video_demo/images'
    # save_path = '/media/dell/data/garbage/video_demo/garbage_seg_demo.mp4'
    img_path = '/media/dell/data/shopout/video_demo/images'
    save_path = '/media/dell/data/shopout/video_demo/出店经营-demo.mp4'
    images2video(img_path, save_path, fps=0.8)
#
#
#
# # 图片路径
# im_dir = 'F:/Projects/auto_Airplane/TS02/truck_tracker/tracker'
# # im_dir = 'E:/LG/GitHub/lg_pro_sets/tmp/generated_labels/000'
# # 输出视频路径
# video_dir = 'util_tmp/video.avi'
# # 帧率
# fps = 24
#
# files = os.listdir(im_dir)
#
#
# # 图片尺寸
# img_size = cv2.imread(os.path.join(im_dir,files[0])).shape[:2][::-1]
#
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# video_writer = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
#
#
# for files in files:
#     im_name = os.path.join(im_dir, files)
#     frame = cv2.imread(im_name)
#     video_writer.write(frame)
#     cv2.imshow('rr', frame)
#     cv2.waitKey(1)
#
# video_writer.release()
# print('finish')
