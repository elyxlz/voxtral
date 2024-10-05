import hashlib
import os
import subprocess
import typing
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import pydantic_settings as pyds
import yt_dlp
from tqdm import tqdm


class ScrapingConfig(pyds.BaseSettings):
    input_file: str = "./data/urls.txt"
    output_path: str = "./data/chunks"
    chunk_duration: int = 20
    max_workers: int = 64
    format: str = "bestaudio[ext=m4a]/best[ext=mp4]/bestaudio/best"


def generate_filename(url: str, chunk_number: int, chunk_size: int) -> str:
    hash_input = f"{url}_{chunk_number}_{chunk_size}"
    hash_object = hashlib.md5(hash_input.encode())
    return str(uuid.UUID(hash_object.hexdigest()))


def download_and_chunk_video(
    url: str, config: ScrapingConfig
) -> typing.Tuple[int, int]:
    print(f"Processing {url}")
    duration = get_video_duration(url)
    if duration == 0:
        print(f"Skipping {url} due to error in fetching duration")
        return 0, 0

    # Download the entire video
    temp_filename = f"temp_{uuid.uuid4()}.%(ext)s"
    ydl_opts = {
        "format": config.format,
        "outtmpl": temp_filename,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            print(f"Successfully downloaded entire video from {url}")
        except Exception as e:
            print(f"Error downloading video from {url}: {str(e)}")
            return 0, 0

    # Get the actual filename after download
    temp_filename = [
        f for f in os.listdir() if f.startswith(temp_filename.split(".")[0])
    ][0]

    # Get the file extension
    _, file_extension = os.path.splitext(temp_filename)

    # Generate a base name for chunks
    base_name = generate_filename(url, 0, config.chunk_duration)

    # Create subdirectory based on the first two letters of the UUID
    subdir = base_name[:2]
    output_subdir = os.path.join(config.output_path, subdir)
    os.makedirs(output_subdir, exist_ok=True)

    # Use FFmpeg to chunk the video
    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        temp_filename,
        "-c",
        "copy",
        "-f",
        "segment",
        "-segment_time",
        str(config.chunk_duration),
        "-reset_timestamps",
        "1",
        "-map",
        "0",
        os.path.join(output_subdir, f"{base_name}_%d{file_extension}"),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Successfully chunked video from {url}")
    except subprocess.CalledProcessError as e:
        print(f"Error chunking video from {url}: {e.stderr.decode()}")
        os.remove(temp_filename)  # Clean up the temporary file in case of error
        return 0, 0

    # Count the number of chunks created
    # Count the number of chunks created
    chunks_downloaded = len(
        [
            f
            for f in os.listdir(output_subdir)
            if f.startswith(base_name) and f.endswith(file_extension)
        ]
    )

    # Remove the temporary file
    os.remove(temp_filename)

    actual_duration = min(chunks_downloaded * config.chunk_duration, duration)
    print(
        f"Finished creating {chunks_downloaded} chunks for {url} (Duration: {actual_duration}s)"
    )
    return chunks_downloaded, actual_duration


def get_video_duration(url: str) -> int:
    print(f"Fetching duration for {url}")
    ydl_opts = {"quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            duration = int(info["duration"])
            print(f"Duration for {url}: {duration} seconds")
            return duration
        except Exception as e:
            print(f"Error fetching duration for {url}: {str(e)}")
            return 0


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def scrape_youtube_urls(config: ScrapingConfig) -> None:
    print("Starting YouTube URL scraping process")

    # Check if FFmpeg is installed
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print(
            "Error: FFmpeg is not installed. Please install FFmpeg before running this script."
        )
        return
    except FileNotFoundError:
        print(
            "Error: FFmpeg is not found in the system PATH. Please install FFmpeg or add it to the PATH."
        )
        return

    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
        print(f"Created output directory: {config.output_path}")
    try:
        with open(config.input_file, "r") as file:
            urls = [line.strip() for line in file if line.strip()]
        print(f"Read {len(urls)} URLs from {config.input_file}")
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return

    total_urls = len(urls)
    total_chunks = 0
    total_duration = 0

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = [
            executor.submit(download_and_chunk_video, url, config) for url in urls
        ]

        with tqdm(
            total=total_urls, desc="Processing URLs", unit="url", colour="yellow"
        ) as pbar:
            for future in as_completed(futures):
                chunks_downloaded, duration = future.result()
                total_chunks += chunks_downloaded
                total_duration += duration
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Total Chunks": total_chunks,
                        "Total Duration": format_duration(total_duration),
                    },
                    refresh=True,
                )

    print("Finished scraping YouTube URLs.")
    print(f"Total chunks downloaded: {total_chunks}")
    print(f"Total duration: {format_duration(total_duration)}")


if __name__ == "__main__":
    print("Script started")
    config = ScrapingConfig()
    scrape_youtube_urls(config)
    print("Script finished")
