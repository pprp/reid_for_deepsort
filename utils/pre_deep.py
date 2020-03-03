import os
import shutil
import cv2
import numpy as np
import glob
import sys

"""[summary]
根据视频和darklabel得到的标注文件
"""


def preprocessVideo(video_path):
    '''
    预处理，将视频变为一帧一帧的图片
    '''
    if not os.path.exists(video_frame_save_path):
        os.mkdir(video_frame_save_path)

    vidcap = cv2.VideoCapture(video_path)
    (cap, frame) = vidcap.read()

    height = frame.shape[0]
    width = frame.shape[1]

    cnt_frame = 0

    while (cap):
        cv2.imwrite(
            os.path.join(video_frame_save_path, "frame_%d.jpg" % (cnt_frame)),
            frame)
        cnt_frame += 1
        print(cnt_frame, end="\r")
        sys.stdout.flush()
        (cap, frame) = vidcap.read()
    vidcap.release()
    return width, height


def postprocess(video_frame_save_path):
    '''
    后处理，删除无用的文件夹
    '''
    if os.path.exists(video_frame_save_path):
        shutil.rmtree(video_frame_save_path)


def extractVideoImgs(frame, video_frame_save_path, coords):
    '''
    抠图
    '''
    x1, y1, x2, y2 = coords
    # get image from save path
    img = cv2.imread(
        os.path.join(video_frame_save_path, "frame_%d.jpg" % (frame)))
    # crop
    save_img = img[y1:y2, x1:x2]
    return save_img


def restrictCoords(width, height, x, y):
    x = max(1, x)
    y = max(1, y)
    x = min(x, width)
    y = min(y, height)
    return x, y


if __name__ == "__main__":

    total_cow_num = 0

    root_dir = "./data/videoAndLabel"
    reid_dst_path = r"./data/reid"

    txt_list = glob.glob(os.path.join(root_dir, "*.txt"))
    video_list = glob.glob(os.path.join(root_dir, "*.mp4"))

    for i in range(len(txt_list)):
        txt_path = txt_list[i]
        video_path = video_list[i]

        print("processing:", video_path)

        if not os.path.exists(txt_path):
            continue

        video_name = os.path.basename(video_path).split('.')[0]
        video_frame_save_path = os.path.join(os.path.dirname(video_path),
                                             video_name)

        f_txt = open(txt_path, "r")

        width, height = preprocessVideo(video_path)

        print("done")

        # video_cow_id = video_name + str(total_cow_num)

        for line in f_txt.readlines():
            bboxes = line.split(',')
            ids = []
            frame_id = int(bboxes[0])

            if frame_id % 30 != 0:
                continue

            num_object = int(bboxes[1])
            for num_obj in range(num_object):
                # obj = 0, 1, 2
                obj_id = bboxes[1 + (num_obj) * 6 + 1]
                obj_x1 = int(bboxes[1 + (num_obj) * 6 + 2])
                obj_y1 = int(bboxes[1 + (num_obj) * 6 + 3])
                obj_x2 = int(bboxes[1 + (num_obj) * 6 + 4])
                obj_y2 = int(bboxes[1 + (num_obj) * 6 + 5])
                # process coord
                obj_x1, obj_y1 = restrictCoords(width, height, obj_x1, obj_y1)
                obj_x2, obj_y2 = restrictCoords(width, height, obj_x2, obj_y2)

                specific_cow_name = video_name + "_" + obj_id

                print("%s:%d-%d-%d-%d" %
                      (specific_cow_name, obj_x1, obj_y1, obj_x2, obj_y2),
                      end='\n')
                # sys.stdout.flush()

                # mkdir for reid dataset
                id_dir = os.path.join(reid_dst_path, specific_cow_name)

                if not os.path.exists(id_dir):
                    os.makedirs(id_dir)

                # save pic
                img = extractVideoImgs(frame_id, video_frame_save_path,
                                       (obj_x1, obj_y1, obj_x2, obj_y2))

                if img is None:
                    print(specific_cow_name + "is empty")
                    continue
                print(frame_id)
                img = cv2.resize(img, (256, 256))

                normalizedImg = np.zeros((256, 256))
                img = cv2.normalize(img, normalizedImg, 0, 255,
                                    cv2.NORM_MINMAX)

                cv2.imwrite(
                    os.path.join(id_dir, "%s_%d.jpg") %
                    (specific_cow_name, frame_id), img)

        f_txt.close()
        postprocess(video_frame_save_path)