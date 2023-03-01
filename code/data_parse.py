import json
import os
import random
import cv2 as cv
from collections import defaultdict

# data_parse aggregates hyperzooms of the same person based on existing tagged datasets
# 1. download appearance-search-dedup dataset and collect matched identities by a monotonically increasing pid 
#    to the bounding box tags specified by the dataset name
# 2. search people-tracking-videos dataset for each pid and the tag, collecting 5 random frames from each annotation
#    and organizing by market1501 format

class Debug:
    def __init__(self):
        self.out = open("/home/georgez/centroids-reid/code/logs.txt", "w")
    def print(self, msg):
        self.out.write(msg + "\n")

class Parser:
    def __init__(self):
        self.match_data = "/home/georgez/datasets/appearance-search-dedup/"
        self.video_data = "/home/georgez/datasets/people-tracking-videos/"
        self.framesperpid = 5
        self.output = "/home/georgez/datasets/verkada_data_limitedquery/"
        self.debug = Debug()
        self.testsplit = 0.2

        os.mkdir(self.output)


    def crop(self, bounding_box, annofile, frame):
        """
        Extract bounding_box pixel area from image frame and return numpyarr of cropped area

        bounding_box: dict of h,w,x,y coordinates taken from annotation
        annofile: annotation file in json format
        frame: numpyarr of image frame
        """
        ht = frame.shape[0]
        wt = frame.shape[1]
        ratio = annofile['image']['height'] / ht

        for i in bounding_box:
            bounding_box[i] /= ratio

        y1 = int(max(0, bounding_box['y']))
        x1 = int(max(0, bounding_box['x']))
        y2 = int(min(ht-y1, bounding_box['h'])) + y1
        x2 = int(min(wt-x1, bounding_box['w'])) + x1

        return frame[y1:y2, x1:x2]


    def extract_frames(self, anno_id, out):
        """
        Takes self.framesperpid number of random frames from specified annotation file and id

        path: folder to save frames to
        filename: annotation file
        aid: annotation id
        pid: person id
        """
        frames_ret = []

        # Open people-tracking-videos annotation file
        filename = self.anno_info[anno_id]['folder']
        pid = self.anno_info[anno_id]['pid']
        cid = self.anno_info[anno_id]['camera']
        vanno_file = open(self.video_data + "releases/latest/annotations/" + filename + ".json")
        vanno_json = json.load(vanno_file)
        
        # Find annotation with aid
        for anno in vanno_json['annotations']:
            if anno['id'] == anno_id:
                frame_anno = anno['frames']
                seg_start = -(anno['segments'][0][0] // -10)
                seg_end = anno['segments'][0][1] // 10

                if seg_end - seg_start > self.framesperpid:
                    frames = random.sample(range(seg_start, seg_end), self.framesperpid) 
                else:
                    frames = range(seg_start, seg_end)

                for i in frames:
                    rpath = str(i) + ".jpg"
                    rpath = self.video_data + "images/" + filename + "/" + rpath

                    frame = cv.imread(rpath)
                    if frame is None:
                        self.debug.print("failed to read from " + filename)
                        continue

                    bounding_box = frame_anno[str(i*10)]['bounding_box']
                    nframe = self.crop(bounding_box, vanno_json, frame)

                    image_path = str(pid) + "_c" + str(cid) + "_" + str(1000000+i)[1:] + ".jpg"
                    cv.imwrite(out + image_path, nframe)
                    frames_ret.append(image_path)

        if len(frames_ret) <= 1:
            for i in frames_ret:
                os.remove(out + i)
                self.debug.print("Deleted pid: " + str(pid))
        elif out == self.test:
            queryimg = random.sample(frames_ret, 1)[0]
            os.system('cp ' + out + queryimg + ' ' + self.query + queryimg)
            os.system('rm ' + out + queryimg)

    def grouping(self):
        """
        Calls extract_frames() on correct file for all tags returned by pid in get_id_match()

        anno_list: 2d list of all grouped annotation tags
        """      

        self.cam2anno = defaultdict(list)
        anno_dir = self.video_data + "releases/latest/annotations/"

        for file in os.listdir(anno_dir):
            cid = file[5:-5]

            with open(anno_dir + file) as anno_file:
                anno_json = json.load(anno_file)

                for anno in anno_json['annotations']:
                    type = anno['name']
                    aid = anno['id']
                    
                    if type == 'person':
                        self.cam2anno[cid].append(aid)
                        if aid not in self.anno_info:
                            self.anno_info[aid] = {'pid': self.pid}
                            self.pid2anno[self.pid] = [aid]
                            self.pid += 1

                        self.anno_info[aid]['folder'] = file[:-5]
                        self.anno_info[aid]['cid'] = cid
        
        length = 0
        longestid = 0
        for pid in self.pid2anno:
            cam = 1
            if len(self.pid2anno[pid]) > length:
                length = len(self.pid2anno[pid])
                longestid = pid

            for anno in self.pid2anno[pid]:
                self.anno_info[anno]['camera'] = cam
                cam += 1
        print("Longest id: " + str(longestid) + " with annos: " + str(length))
        
        camids = self.groupingHelper(longestid)
        self.train = self.output + "bounding_box_train/"
        self.test = self.output + "bounding_box_test/"
        self.query = self.output + "query/"
        os.mkdir(self.train)
        os.mkdir(self.test)
        os.mkdir(self.query)
        print("Cameras in test, train: " + str(len(camids)) + ", " + str(len(self.cam2anno)-len(camids)))

        with open('evalanno.txt', 'w') as evalfile, open('trainanno.txt', 'w') as trainfile:
            for cid in self.cam2anno:
                for anno in self.cam2anno[cid]:
                    if anno in self.pid2anno[longestid]:
                        continue
                    if cid in camids:
                        evalfile.write(anno+'\n')
                        self.extract_frames(anno, self.test)
                    else:
                        self.extract_frames(anno, self.train)
                        trainfile.write(anno+'\n')


    def groupingHelper(self, longestid):
        test_cams = set()
        all_cams = set(self.cam2anno.keys())
        threshold_met = False
        count = 0
        threshold = int(len(self.anno_info) * self.testsplit)
        print("Total number of annotations: " + str(len(self.anno_info)))
        print("Total number of identities: " + str(self.pid))

        while not threshold_met:
            cur_cams = set([all_cams.pop()])
            
            while cur_cams:
                cur_cam = cur_cams.pop()
                test_cams.add(cur_cam)
                count += len(self.cam2anno[cur_cam])

                for anno in self.cam2anno[cur_cam]:
                    if anno in self.pid2anno[longestid]:
                        count -= 1
                        continue

                    pid = self.anno_info[anno]['pid']
                    for panno in self.pid2anno[pid]:
                        try:
                            cam = self.anno_info[panno]['cid']
                        except:
                            continue
                        if cam in all_cams:
                            all_cams.remove(cam)
                            cur_cams.add(cam)
            
            if count < threshold + 500 and count > threshold - 500:
                threshold_met = True
            elif count > threshold + 500:
                print(count)
                print(threshold)
                count = 0
                all_cams = set(self.cam2anno.keys())
                test_cams = set()

        print("Approx frames in test, train: " + str(count * 5) + ", " + str((len(self.anno_info)-count)*5))
        return test_cams
        

    def parse(self):
        """
        Reads all annotation files from appearance-search-dedup and returns dictionary of 
        (int) pid: [list of tagged annotation ids]
        """

        pid2anno = {}
        anno2pid = {}
        pid = 0
        anno_dir = self.match_data + "releases/latest/annotations/"
        random.seed(10)

        for filename in sorted(os.listdir(anno_dir)):
            with open(anno_dir + filename) as anno_file:
                anno_json = json.load(anno_file)
                anno = anno_json['annotations']
                
                if not anno:
                    continue
                anno = anno[0]
                
                matches = filename.split("_")
                first, second = matches[0], matches[1][:-5]

                if anno['name'] == "Same":
                    # Clustering conditions: any matches will be clustered with previous identities
                    if first not in anno2pid and second not in anno2pid:
                        pid2anno[pid] = [first, second]
                        anno2pid[first] = pid
                        anno2pid[second] = pid
                        pid += 1

                    elif first in anno2pid and second not in anno2pid:
                        tmp = anno2pid[first]
                        pid2anno[tmp].append(second)
                        anno2pid[second] = tmp

                    elif second in anno2pid and first not in anno2pid:
                        tmp = anno2pid[second]
                        pid2anno[tmp].append(first)
                        anno2pid[first] = tmp

                    elif anno2pid[first] != anno2pid[second]:
                        # # EXTRACT ON MATCH
                        # pid2anno[anno2pid[first]].append(second)
                        # pid2anno[anno2pid[second]].remove(second)
                        # anno2pid[second] = anno2pid[first]

                        # MERGE ON MATCH
                        tmp = pid2anno[anno2pid[second]]
                        pid2anno[anno2pid[first]].extend(tmp)
                        pid2anno.pop(anno2pid[second])
                        for i in tmp:
                            anno2pid[i] = anno2pid[first]

                else:
                    if first not in anno2pid:
                        anno2pid[first] = pid
                        pid2anno[pid] = [first]
                        pid += 1
                    if second not in anno2pid:
                        anno2pid[second] = pid
                        pid2anno[pid] = [second]
                        pid += 1
                    # if anno2pid[first] == anno2pid[second]:
                    #     pid2anno[anno2pid[second]].remove(second)
                    #     pid2anno[pid] = [second]
                    #     anno2pid[second] = pid
                    #     pid += 1

        ret = list(pid2anno.values())
        self.anno_info = {}
        self.pid2anno = {}
        self.pid = 1
        for i,j in enumerate(ret):
            for anno in j:
                self.anno_info[anno] = {'pid': self.pid}
            if j:
                self.pid2anno[self.pid] = j
                self.pid += 1
        
        print("Number of matched tags: " + str(len(self.anno_info)))
        print("Number of identities: " + str(self.pid))
        self.grouping()
        
parser = Parser()
parser.parse()

