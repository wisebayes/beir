import os
import requests
import zipfile


def download_url(url, save_path, chunk_size=128):
    if not os.path.isfile(save_path):
        r = requests.get(url, stream=True)
        with open(save_path, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=chunk_size):
                fd.write(chunk)

def unzip(zip_file, out_dir):
    if not os.path.isdir(zip_file.replace(".zip", "")):
        zip_ = zipfile.ZipFile(zip_file, "r")
        zip_.extractall(path=out_dir)
        zip_.close()

def download_and_unzip(url, out_dir):
    
    os.makedirs(out_dir, exist_ok=True)
    dataset = url.split("/")[-1]
    zip_file = os.path.join(out_dir, dataset)
    
    print("Downloading {} ...".format(dataset))
    download_url(url, zip_file)
    
    print("Unzipping {} ...".format(dataset))
    unzip(zip_file, out_dir)
    
    return os.path.join(out_dir, dataset.replace(".zip", ""))