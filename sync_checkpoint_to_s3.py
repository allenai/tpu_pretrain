import boto3
import shutil
import glob
import sys
import os
import re
from tqdm import tqdm
import argparse


s3_resource = boto3.resource("s3", region_name="us-east-1")

def sync(local_dir, keypath):
    try:
        my_bucket = s3_resource.Bucket("suching-dev")
        for path, subdirs, files in os.walk(local_dir):
            path = path.replace("\\","/")
            directory_name = keypath
            for file in tqdm(files):
                my_bucket.upload_file(os.path.join(path, file), keypath + "/" + file)

    except Exception as err:
        print(err)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    model = sys.argv[1]
    checkpoint = sys.argv[2]
    dest_dir = "/tmp/roberta-base-ft"
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    for file in glob.glob(f'{model}/*.json'):
        shutil.copy(file, dest_dir)
    shutil.copy(f'{model}/merges.txt', dest_dir)
    shutil.copy(f'{model}/pytorch_model-{checkpoint}.bin', dest_dir + "/pytorch_model.bin")
    sync(dest_dir, f"roberta-checkpoints/checkpoint-{checkpoint}")
    shutil.rmtree(dest_dir)