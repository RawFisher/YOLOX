import os
import numpy as np
import json
import shutil
import xml.dom.minidom as xml
import abc


# Use the same script for MOT16
DATA_PATH = 'datasets/mot'
OUT_PATH = os.path.join(DATA_PATH, 'annotations')
SPLITS = ['train', 'test']
HALF_VIDEO = False
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False


'''
读取xml文件
'''

class XmlReader(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        pass
    def read_content(self,filename):
        content = None
        if (False == os.path.exists(filename)):
            return content
        filehandle = None
        try:
            filehandle = open(filename,'rb')
        except FileNotFoundError as e:
            print(e.strerror)
        try:
            content = filehandle.read()
        except IOError as e:
            print(e.strerror)
        if (None != filehandle):
            filehandle.close()
        if(None != content):
            return content.decode("utf-8","ignore")
        return content

    @abc.abstractmethod
    def load(self,filename):
        pass

class XmlTester(XmlReader):
    def __init__(self):
        XmlReader.__init__(self)
    def load(self, filename):
        filecontent = XmlReader.read_content(self,filename)
        #print(filecontent)
        seq_gt=[]

        if None != filecontent:
            dom = xml.parseString(filecontent)
            root = dom.getElementsByTagName('sequence')[0]
            if root.hasAttribute("name"):
                seq_name=root.getAttribute("name")
                print ("*"*20+"sequence: %s" %seq_name +"*"*20)
            #获取所有的frame
            frames = root.getElementsByTagName('frame')

            for frame in frames:
                if frame.hasAttribute("num"):
                    frame_num=int(frame.getAttribute("num"))

                    # print ("-"*10+"frame_num: %s" %frame_num +"-"*10)

                target_list = frame.getElementsByTagName('target_list')[0]
                #获取一帧里面所有的target
                targets = target_list.getElementsByTagName('target')
                targets_dic={}
                for target in targets:
                    if target.hasAttribute("id"):
                        tar_id=int(target.getAttribute("id"))
                        # print ("id: %s" % tar_id)

                    box = target.getElementsByTagName('box')[0]
                    if box.hasAttribute("left"):
                        left=box.getAttribute("left")
                        #print ("  left: %s" % left)
                    if box.hasAttribute("top"):
                        top=box.getAttribute("top")
                        #print ("  top: %s" %top )
                    if box.hasAttribute("width"):
                        width=box.getAttribute("width")
                        #print ("  width: %s" % width)
                    if box.hasAttribute("height"):
                        height=box.getAttribute("height")
                        #print ("  height: %s" %height )
                    #中心坐标
                    x=float(left)+float(width)/2
                    y=float(top)+float(height)/2
                    #宽高中心坐标归一化
                    # x/=img_w
                    # y/=img_h
                    # width=float(width)/img_w
                    # height=float(height)/img_h

                    attribute = target.getElementsByTagName('attribute')[0]
                    if attribute.hasAttribute("vehicle_type"):
                        type=attribute.getAttribute("vehicle_type")
                        if type=="car":
                            type=0
                        if type=="van":
                            type=1
                        if type=="bus":
                            type=2
                        if type=="others":
                            type=3

                    #anno_f.write(str(type)+" "+tar_id+" %.3f"%x+" %.3f"%y+" %.3f"%width+" %.3f"%height+"\n")
                    seq_gt.append([frame_num,tar_id,x,y,float(width),float(height),type])
        return seq_gt


if __name__ == '__main__':

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    for split in SPLITS:
        if split == "test":
            img_path = os.path.join(DATA_PATH, 'Insight-MVT_Annotation_Test')
            anno_path = os.path.join(DATA_PATH, 'DETRAC-Test-Annotations-XML')
        else:
            img_path = os.path.join(DATA_PATH, 'Insight-MVT_Annotation_Train')
            anno_path = os.path.join(DATA_PATH, 'DETRAC-Train-Annotations-XML')
        out_path = os.path.join(OUT_PATH, '{}.json'.format(split))
        out = {'images': [], 'annotations': [], 'videos': [],
               'categories': [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'van'},
                              {'id': 3, 'name': 'bus'}, {'id': 4, 'name': 'others'}]}
        seqs = os.listdir(img_path)
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        tid_curr = 0
        tid_last = -1
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            video_cnt += 1  # video sequence number.
            out['videos'].append({'id': video_cnt, 'file_name': seq})
            seq_img_path = os.path.join(img_path, seq)
            seq_anno_path = os.path.join(anno_path, seq+'.xml')
            images = os.listdir(seq_img_path)
            num_images = len([image for image in images if 'jpg' in image])  # half and half

            for i in range(num_images):
                height, width = 540, 960
                image_info = {'file_name': '{}/img{:05d}.jpg'.format(seq, i + 1),  # image name.
                              'id': image_cnt + i + 1,  # image number in the entire training set.
                              'frame_id': i + 1,  # image number in the video sequence, starting from 1.
                              'prev_image_id': image_cnt + i if i > 0 else -1,  # image number in the entire training set.
                              'next_image_id': image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt,
                              'height': height, 'width': width}
                out['images'].append(image_info)


            reader = XmlTester()
            gt = reader.load(seq_anno_path)

            ids = []
            for line in gt:
                if not line[1] in ids:
                    ids.append(line[1])
            ids = sorted(ids)
            print(ids)
            print('len(ids)=', len(ids))

            final_gt = []
            for id in ids:
                for line in gt:
                    if line[1] == id:
                        final_gt.append(line)

            # label_fpath = os.path.join(img_path, seq, 'gt')
            # if not os.path.exists(label_fpath):
            #     os.mkdir(label_fpath)
            # label_fpath = os.path.join(label_fpath, 'gt.txt')
            for fid, tid, x, y, w, h, label in final_gt:
                ann_cnt += 1
                label = int(label)
                frame_id = int(fid)
                track_id = int(tid)
                x, y, w, h = float(x), float(y), float(w), float(h)
                if not track_id == tid_last:
                    tid_curr += 1
                    tid_last = track_id
                ann = {'id': ann_cnt,
                       'category_id': label + 1,
                       'image_id': image_cnt + frame_id,
                       'track_id': tid_curr,
                       'bbox': [x-w/2, y-h/2, w, h],
                       'conf': 1.0,
                       'iscrowd': 0,
                       'area': float(w * h),
                       'video_id': video_cnt}
                out['annotations'].append(ann)
                # TODO: visibility maybe less 1
                label_str = '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d}\n'.format(frame_id, tid_last,
                                                                                    int(x-w/2), int(y-h/2), int(w), int(h), 1, label, 1)
                # with open(label_fpath, 'a') as f:
                #     f.write(label_str)
            image_cnt += num_images
            print(tid_curr, tid_last)
        print('loaded {} for {} images and {} samples'.format(split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))