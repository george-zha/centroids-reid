import boto3
from boto3.dynamodb.conditions import Key, Attr
import botocore.exceptions
import os
import pandas as pd
import shutil
import time
import zipfile

### hz_download script for downloading past day hyperzooms from verkada demo cameras --> found in vcamera_public_cameras
### 1. queries hyperzoom metadata from dynamoDB 
### 2. download batched hyperzooms from s3 based on metadata
### 3. unzip and compile in hz_demo folder

PERSON = 'person'
BLOBLOCATION = 'blobLocation'
ORIGKEY = 'origKey'
BUCKET = 'bucket'
DOWNLOADPATH = '/home/george/datasets/hyperzoom_data/'
TMPFOLDER = '/home/george/mtmp/'

s3 = boto3.resource('s3')
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('vsubmit_device_timeseries.prod1')


def hz_by_device(d_ids):
    """
    Given deviceid d_id, find devices with hyperzooms within the past day and save blobLocation

    d_ids: list of device ids
    """
    
    ret = {}
    timethresh = (int(time.time()) - 86400) * 1000
    print("Time filter: " + str(timethresh))

    for id in d_ids:
        response = table.query(KeyConditionExpression = Key('deviceId').eq(id) & Key('recordTypeTimeMsUid').gt('HyperzoomRecord_' + str(timethresh)),
            FilterExpression = Attr('detectedClass').eq(PERSON))
        items = response['Items']

        while response.get('LastEvaluatedKey'):
            response = table.query(KeyConditionExpression = Key('deviceId').eq(id) & Key('recordTypeTimeMsUid').gt('HyperzoomRecord_' + str(timethresh)),
            FilterExpression = Attr('detectedClass').eq(PERSON), ExclusiveStartKey = response['LastEvaluatedKey'])
            items.extend(response['Items'])

        if items:
            loc = [None] * len(items)
            for i,j in enumerate(items):
                # blobLocation saved as batched_hz / 4 digit code / random folder id
                loc[i] = {BLOBLOCATION: j[BLOBLOCATION], ORIGKEY: j[ORIGKEY], BUCKET: j[BUCKET]}

            ret[id] = loc
            
            print("Got id: " + str(id) + " with " + str(len(loc)) + " entries")

    print("Number of devices: " + str(len(ret)))
    bucket_download(ret)


def bucket_download(bloblocs):
    """
    Takes device_id, location pairings and downloads applicable HZs
    bloblocs: {device_id, list of (blobLocation, origKey within blob)}
    """
    bucket = s3.Bucket(BUCKET)

    for device_id in bloblocs:
        os.mkdir(TMPFOLDER)
        for loc in bloblocs[device_id]:
            blobloc = loc[BLOBLOCATION]
            origkey = loc[ORIGKEY]
            bucket = s3.Bucket(loc[BUCKET])

            path = blobloc.split("/")
            path = "_".join(path[1:])

            try:
                bucket.download_file(blobloc, TMPFOLDER + path)
            except botocore.exceptions.ClientError as error:
                print(error)
                continue
            try:
                with zipfile.ZipFile(TMPFOLDER + path, 'r') as zip_ref:
                    zip_ref.extractall(TMPFOLDER)
            except zipfile.BadZipFile as error:
                print("Not a zip!")
                continue
            
            filename = origkey.split("/")
            filename = filename[0] + "_" + filename[2]
            shutil.copy(TMPFOLDER + origkey, DOWNLOADPATH + filename)
        shutil.rmtree(TMPFOLDER)
        print("Downloaded all from device: " + device_id)

camera_csv = pd.read_csv('/home/george/vcamera_public_cameras.csv', usecols=[0], header=None)
d_ids = camera_csv[0].values
hz_by_device(d_ids)
