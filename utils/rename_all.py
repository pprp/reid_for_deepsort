import os
import shutil
import os.path as osp

root_dir = r'C:\Users\pprp\Desktop\face\head_test'

for i in os.listdir(root_dir):
    new_dir = osp.join(root_dir, i)
    for j in os.listdir(new_dir):
        jpg = osp.join(new_dir, j)
        name, frame, head = j.split("_")
        frame_no = int(frame)
        extend_no = '%04d' % frame_no
        newName = 'head_' + name + "_" + '%s.jpg' % (str(extend_no))
        print('from %s to %s' % (j, newName))
        os.rename(jpg, os.path.join(new_dir, newName))