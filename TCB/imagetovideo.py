
import cv2 as cv
import os

def image_to_video():
    file_path = 'vis/MOT17-04-FRCNN/'  # 图片目录
    output = 'output.mp4'  # 生成视频路径
    img_list = os.listdir(file_path)  # 生成图片目录下以图片名字为内容的列表
    num_images = len([image for image in img_list if 'jpg' in image])
    height = 1080
    weight = 1920
    fps = 30
    # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G') 用于avi格式的生成
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 用于mp4格式的生成
    videowriter = cv.VideoWriter(output, fourcc, fps, (weight, height))  # 创建一个写入视频对象
    for i in range(num_images):
        print(i)
        path = os.path.join(file_path,"{}.jpg".format(i+1))
        frame = cv.imread(path)
        videowriter.write(frame)

    videowriter.release()

image_to_video()
