from kaggle.api.kaggle_api_extended import KaggleApi
import os


def dataset_download(dataset_url):
    output_dir = "dataset/"

    if os.path.exists(output_dir):
        for f in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, f))
    else:
        os.makedirs(output_dir)

    # Descargar desde Kaggle
    kaggleApi = KaggleApi()
    kaggleApi.authenticate()
    print(f"🗃️ Downloading from Kaggle: {dataset_url}")
    kaggleApi.dataset_download_files(dataset_url, path=output_dir, unzip=True)
    print("✅ Download completed")
    
    return os.path.join(output_dir, [f for f in os.listdir(output_dir) if f.endswith(".csv")][0])
