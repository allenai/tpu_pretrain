import boto3
import shutil
import glob
import sys
import os
import re
from tqdm import tqdm

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
    model = sys.argv[1]
    s3_dest = sys.argv[2]
    dest_dir = "/tmp/roberta-base-ft"
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    for file in glob.glob(f'{model}/*.json'):
        shutil.copy(file, dest_dir)
    shutil.copy(f'{model}/merges.txt', dest_dir)
    checkpoints = glob.glob(f'{model}/*.bin') # * means all if need specific format then *.csv
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    checkpoint_index = re.findall(r'\d{4}', latest_checkpoint)[0]
    shutil.copy(f'{model}/{os.path.basename(latest_checkpoint)}', dest_dir + "/pytorch_model.bin")
    sync(dest_dir, f"{s3_dest}/roberta-{checkpoint_index}")
    shutil.rmtree(dest_dir)