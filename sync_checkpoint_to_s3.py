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
                my_bucket.upload_file(os.path.join(path, file), f"{keypath}/{file}")

    except Exception as err:
        print(err)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    model = sys.argv[1]
    checkpoint = sys.argv[2]
    tmp_dir = "/tmp/roberta-base-ft"
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    for file in glob.glob(f'{model}/*.json'):
        shutil.copy(file, tmp_dir)
    shutil.copy(f'{model}/merges.txt', tmp_dir)
    shutil.copy(f'{model}/pytorch_model-{checkpoint}.bin', tmp_dir + "/pytorch_model.bin")
    sync(tmp_dir, f"roberta-checkpoints/{model}/checkpoint-{checkpoint}")
    shutil.rmtree(tmp_dir)