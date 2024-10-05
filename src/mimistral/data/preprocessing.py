import os
import uuid
import hashlib
import typing
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

import yt_dlp
from tqdm import tqdm


class DownloaderConfig(typing.NamedTuple):
    input_file: str
    output_path: str
    chunk_duration: int
    max_workers: int = 5
    format: str = "bestaudio/best"
    preferred_format: str = "mp4"


def read_urls_from_file(file_path: str) -> list[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


def generate_filename(url: str, chunk_number: int, chunk_size: int) -> str:
    hash_input = f"{url}_{chunk_number}_{chunk_size}"
    hash_object = hashlib.md5(hash_input.encode())
    return str(uuid.UUID(hash_object.hexdigest()))


def download_video(url: str, config: DownloaderConfig, chunk_number: int) -> bool:
    filename = generate_filename(url, chunk_number, config.chunk_duration)
    output_template = os.path.join(config.output_path, f"{filename}.%(ext)s")

    ydl_opts = {
        "format": config.format,
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegVideoConvertor",
                "preferedformat": config.preferred_format,
            }
        ],
        "postprocessor_args": [
            "-ss",
            str(chunk_number * config.chunk_duration),
            "-t",
            str(config.chunk_duration),
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            return True
        except Exception as e:
            print(f"Error downloading {url}: {str(e)}")
            return False


def scrape_youtube_urls(urls: list[str], config: DownloaderConfig) -> None:
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for url in urls:
            chunk_number = 0
            while True:
                future = executor.submit(download_video, url, config, chunk_number)
                futures.append(future)
                chunk_number += 1

                # Check if we've reached the end of the video
                if not future.result():
                    break

        with tqdm(total=len(futures), desc="Downloading chunks") as pbar:
            for future in as_completed(futures):
                pbar.update(1)


if __name__ == "__main__":
    config = DownloaderConfig(
        input_file="urls.txt",
        output_path="downloaded_chunks",
        chunk_duration=10,  # Duration of each chunk in seconds
        max_workers=5,  # Number of concurrent downloads
    )

    urls = read_urls_from_file(config.input_file)
    scrape_youtube_urls(urls, config)
