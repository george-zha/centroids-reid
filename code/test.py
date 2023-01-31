import json
import os
import random
import numpy as np
import cv2 as cv

MATCHDATA = "/home/george/datasets/appearance-search-dedup/"
VIDEODATA = "/home/george/datasets/people-tracking-videos/"
#VIDEODATA = "/home/songcao/datasets/people-tracking-videos"
FRAMESPERVID = 5
SAVELOC = "/home/george/datasets/verkada_data/"

def get_id_match():
    """
    Reads all annotation files from appearance-search-dedup and returns dictionary of 
    (int) pid: [list of tagged annotation ids]
    """
    tagbypid = {}
    pidbytag = {}
    pidtohistory = {}
    pid = 0
    anno_dir = MATCHDATA + "/releases/latest/annotations/"

    for filename in os.listdir(anno_dir):
        with open(anno_dir + filename) as anno_file:
            anno_json = json.load(anno_file)
            
            for anno in anno_json['annotations']:
                if anno['name'] == "Same":
                    matches = filename.split("_")
                    first, second = matches[0], matches[1][:-5]

                    # Clustering conditions: any matches will be clustered with previous identities
                    if first not in pidbytag and second not in pidbytag:
                        tagbypid[pid] = [first, second]
                        pidtohistory[pid] = [filename]
                        pidbytag[first] = pid
                        pidbytag[second] = pid
                        
                        pid += 1

                    elif first in pidbytag and second not in pidbytag:
                        tmp = pidbytag[first]
                        tagbypid[tmp].append(second)
                        pidtohistory[tmp].append(filename)
                        pidbytag[second] = tmp

                    elif second in pidbytag and first not in pidbytag:
                        tmp = pidbytag[second]
                        tagbypid[tmp].append(first)
                        pidtohistory[tmp].append(filename)
                        pidbytag[first] = tmp

                    elif pidbytag[first] != pidbytag[second]:
                        tmp = tagbypid[pidbytag[second]]
                        tagbypid.pop(pidbytag[second])
                        tagbypid[pidbytag[first]].extend(tmp)
                        pidtohistory[pidbytag[first]].extend(pidtohistory[pidbytag[second]])
                        for i in tmp:
                            pidbytag[i] = pidbytag[first]
                    
    print("Number of identities: " + str(len(tagbypid)))
    print("Number of matched tags: " + str(len(pidbytag)))

    longest = 0
    longestind = 0
    for i in tagbypid:
        if len(tagbypid[i]) > longest:
            longestind = i
            longest = len(tagbypid[i])
    print(longestind)
    print(len(tagbypid[longestind]))


    testdir = SAVELOC+"test/"
    compdir = MATCHDATA+"images/"
    os.mkdir(testdir)
    for i in pidtohistory[longestind]:
        os.system('cp ' + compdir+i[:-5]+".jpg" + ' ' + testdir+i[:-5]+".jpg")
        

get_id_match()