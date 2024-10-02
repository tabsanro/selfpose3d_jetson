import gdown
import os
import os.path as osp

# Download the POC dataset and model file from Google Drive

PWD = osp.dirname(osp.abspath(__file__))

# dataset google drive file id
video_ids = {
    'hd_00_01.mp4': '1exSVoAYuEL6EJA2x2rjtHO9vqD6IZj3o',
    'hd_00_02.mp4': '1fBzgSuxfSNCwTxNv-Jyn26U6HmsIm-ms',
    'hd_00_03.mp4': '1-8zncn5IzS17ojfObnryyaHxrkt78aHX',
    'hd_00_04.mp4': '1KdTKLFV67PnfAGk9ePeUGGwdRIyn0De8'
}

calibration_ids = {
    'camera1.pkl': '1YN43TzV-KprwBs2m4dl2pUfxdfSeqom5',
    'camera2.pkl': '17xVFyCikpykltceH6PfdNR5Vgl9rNxQp',
    'camera3.pkl': '1rM6SmUdzRjFcaAjuus7ujYqelxfdhLFY',
    'camera4.pkl': '1H4II88TPcI0y4z0uMDSaJ2lKFCQws9zi'
}

model_ids = {
    'POC_posenet.pth.tar': '1w-7HwZSFcKt6SioETopjA7Fd4-OQEO-I',
    'backbone_32.engine': '1NoJqA8JGRVqq7NBTZQQcSOYBhfrjsjF_',
    'backbone.engine': '1pwwsMJ0oWr-izJ1oss1tMLZMddkCdJlI'  
}

def download_from_google_drive(file_id, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def run():
    for video_name, video_id in video_ids.items():
        output_path = osp.join(PWD, '..', 'data_0705', 'hdVideos', video_name)
        if osp.exists(output_path):
            print(f"{video_name} already exists")
            continue
        download_from_google_drive(video_id, output_path)
        print(f"Downloaded {video_name}")

    for calibration_name, calibration_id in calibration_ids.items():
        output_path = osp.join(PWD, '..', 'data_0705', 'calibration', calibration_name)
        if osp.exists(output_path):
            print(f"{calibration_name} already exists")
            continue
        download_from_google_drive(calibration_id, output_path)
        print(f"Downloaded {calibration_name}")

    for model_name, model_id in model_ids.items():
        output_path = osp.join(PWD, '..', 'models', model_name)
        if osp.exists(output_path):
            print(f"{model_name} already exists")
            continue
        download_from_google_drive(model_id, output_path)
        print(f"Downloaded {model_name}")

    print("Download completed")

if __name__ == '__main__':
    run()