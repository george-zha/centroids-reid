import argparse
import boto3
from botocore.exceptions import ClientError
import logging
import os
from time import time
import torch
import sys
import numpy as np
import shutil

sys.path.append(".")

from config import cfg
from train_ctl_model import CTLModel
from person_attributes_v2.models.AttrLitModel import AttrLitModel
from datasets.transforms import ReidTransforms
from utils.reid_metric import (
    cosine_similarity,
    get_euclidean
)
from inference_utils import (
    ImageDataset,
    make_inference_data_loader,
    pil_loader,
    attr_dataloader
)

CONFIG = '/home/georgez/centroids-reid/configs/256_resnet50_inference.yml'
GALLERY_DATA = '/home/georgez/datasets/hyperzoom_data/'
boto3.setup_default_session(profile_name='prod1')

class SplitSoftMax(torch.nn.Module):
    def __init__(self, splits=[2,10,10,2]):
        super().__init__()
        self.splits = splits
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        ''' split inputs (logits)'''
        preds = x.split(self.splits, dim=1)
        return torch.cat([self.softmax(pred)for pred in preds], dim=1)

class SearchPipeline:
    """
    Pipeline containing functions to index and query embeddings
    """
    def __init__(self, threshold):
        self.s3_client = boto3.client('s3')
        self.bucket = 'verkada-cv-datasets'
        self.embed_folder = 'appearance-search-demo/embeddings/'
        self.attr_embed_folder = 'appearance-search-demo/attr_embeddings/'
        self.paths_folder = 'appearance-search-demo/paths/'
        self.threshold = threshold

        if not os.path.exists('./tmp/'):
            os.mkdir('./tmp/')
        self.embedpath = './tmp/tmp.npy'
        self.a_embedpath = './tmp/tmpattr.npy'
        self.filepath = './tmp/tmppath.npy'
        self.cfg = cfg
        cfg.merge_from_file(CONFIG)

        self.extract_pid = (lambda x: (x).rsplit(".", 1)[0].rsplit("_", 1)[0])
        self.model = CTLModel.load_from_checkpoint(cfg.MODEL.PRETRAIN_PATH)
        self.attr_model = AttrLitModel.load_from_checkpoint('/home/georgez/batch_balanced_resnet152.ckpt')
        self.attr_model = torch.nn.Sequential(self.attr_model, SplitSoftMax(splits=[2,10,10,2]))

        transforms_base = ReidTransforms(self.cfg)
        self.transforms = transforms_base.build_transforms(is_train=False)
        logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
        self.log = logging.getLogger(__name__)

        self.log.info(f"Preparing gallery data")
        self.gallery_loader = make_inference_data_loader(self.cfg, GALLERY_DATA, ImageDataset)

        if len(self.gallery_loader) == 0:
            raise RuntimeError("Length of dataloader = 0")  

    def save_embeddings(self):
        """
        Generate embeddings and save to s3
        """
        for idx, batch in enumerate(self.gallery_loader):
            print("start")
            delta_start = time()
            data, attr_data, filenames = batch
            embeddings = self.model(data)
            embeddings = torch.nn.functional.normalize(embeddings, dim=1, p=2)

            np.save(self.embedpath, embeddings)
            np.save(self.filepath, filenames)

            try:
                self.s3_client.upload_file(self.embedpath, self.bucket, self.embed_folder+str(idx))
            except ClientError:
                self.log.info(ClientError)
                self.log.info("Failed to upload batch {idx}")
                continue
            try:
                self.s3_client.upload_file(self.filepath, self.bucket, self.paths_folder+str(idx))
            except:
                self.log.info("Failed to upload batch {idx} filenames")
                self.s3_client.delete_object(self.bucket, self.embed_folder+str(idx))
            with torch.no_grad():
                embeddings = self.attr_model(attr_data)
            q_attr_feats = list(torch.tensor_split(embeddings, [2,12,22], dim=1))

            for x,feat in enumerate(q_attr_feats):
                q_attr_feats[x] = torch.nn.functional.normalize(feat, dim=1, p=2)
            norm_embed = np.asarray(q_attr_feats)
            np.save(self.a_embedpath, norm_embed)
            self.s3_client.upload_file(self.a_embedpath, self.bucket, self.attr_embed_folder+str(idx))

            print("Batch {idx} took " + str(time() - delta_start))

        os.remove(self.a_embedpath)
        os.remove(self.embedpath)
        os.remove(self.filepath)
    
    def query_embeddings(self, img_path):
        """
        Nearest neighbor search for image located at img_path
        """
        img = pil_loader(img_path)
        img = self.transforms(img)
        query_embed = self.model(torch.unsqueeze(img, dim=0))
        query_embed = torch.nn.functional.normalize(query_embed, dim=1, p=2)

        embed_paths = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.embed_folder)
        filename_paths = self.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.paths_folder)
        assert (len(embed_paths) == len(filename_paths))
        
        results = []
        for key in embed_paths['Contents']:
            idx = key['Key'].split('/')[-1]
            self.s3_client.download_file(self.bucket, self.embed_folder+idx, self.embedpath)
            self.s3_client.download_file(self.bucket, self.paths_folder+idx, self.filepath)
        
            embeds = torch.from_numpy(np.load(self.embedpath))
            paths = np.load(self.filepath)

            distmat = get_euclidean(x=query_embed, y=embeds).cpu().numpy()
            indices = np.argsort(distmat, axis=1)
            
            for idx in indices[0]:
                if distmat[0][idx] < self.threshold:
                    img_path = paths[idx]
                    results.append((img_path, distmat[0][idx]))
                else:
                    break

        os.remove(self.embedpath)
        os.remove(self.filepath)
        return results
    
    def measure_performance(self):
        """
        Calculate time to generate embeddings for percentiles measured in milliseconds
        """
        embeddings = [None for i in range(100)]
        paths = [None for i in range(100)]
        latency = []

        for idx, elem in enumerate(self.gallery_loader):
            if idx == 100:
                self.log.info(f"Number of processed images: {idx}")
                break
            delta_start = time()
            embeddings[idx], paths[idx] = self.model(elem)
            delta = time()
            latency.append(delta-delta_start)

        print("number of iterations: " + str(len(embeddings)))
        print("Latency P50: {:.0f}".format(np.percentile(latency, 50)*1000.0))
        print("Latency P90: {:.0f}".format(np.percentile(latency, 90)*1000.0))
        print("Latency P95: {:.0f}".format(np.percentile(latency, 95)*1000.0))
        print("Latency P99: {:.0f}\n".format(np.percentile(latency, 99)*1000.0))

        return embeddings, paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index and query embeddings from s3"
    )
    parser.add_argument(
        "--save_embeddings", default=False, help="whether to save embeddings to s3", type=bool
    )
    parser.add_argument(
        "--output_dir", default='./results/', type=str
    )
    parser.add_argument(
        "--threshold", default=0.5, type=float
    )
    parser.add_argument(
        "--query", default="", type=str
    )
    args = parser.parse_args()

    test = SearchPipeline(args.threshold)
    if args.save_embeddings: test.save_embeddings()
    # matches = sorted(test.query_embeddings(args.query), key=lambda x : x[1], reverse=True)

    # try:
    #     shutil.rmtree(args.output_dir)
    # except:
    #     pass
    # try:
    #     os.mkdir(args.output_dir)
    # except:
    #     pass
    # for i in matches:
    #     os.system('cp ' + i[0] + ' ' + args.output_dir + str(i[1]) + '.jpg')
    #     print(i)
