import os
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def download_video(url, output_path, chunk_duration):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(id)s_%(title)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mp4",
            }
        ],
        "postprocessor_args": [
            "-ss",
            "0",
            "-t",
            str(chunk_duration),
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False


def process_txt(txt_file, output_folder, chunk_duration, max_workers=5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(txt_file, "r") as file:
        urls = [line.strip() for line in file if line.strip()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(download_video, url, output_folder, chunk_duration)
            for url in urls
        ]

        with tqdm(total=len(urls), desc="Downloading videos") as pbar:
            for future in as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    urls = "urls.txt"
    output_folder = "downloaded_videos"
    chunk_duration = 10  # Duration of each chunk in seconds
    max_workers = 5  # Number of concurrent downloads

    process_txt(urls, output_folder, chunk_duration, max_workers)
