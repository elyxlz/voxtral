import concurrent.futures
from youtubesearchpython import VideosSearch
from tqdm import tqdm
import time
import typing
import os
from rich import print as rprint


class IndexConfig(typing.NamedTuple):
    input_file: str = "./data/searches.txt"
    output_file: str = "./data/urls.txt"
    min_duration: int = 30 * 60
    max_retries: int = 3
    search_limit: int = 30
    retry_delay: float = 2.0
    max_workers: int = 32
    progress_bar: bool = True
    use_multithreading: bool = True
    multithreaded_sleep: float = 3


def search_youtube(
    query: str,
    min_duration: int,
    max_retries: int,
    search_limit: int,
    retry_delay: float,
) -> list[tuple[str, int]]:
    for attempt in range(max_retries):
        try:
            search = VideosSearch(query, limit=search_limit)
            results: list[tuple[str, int]] = []
            while len(results) < search_limit:
                search_result = search.result()
                if search_result is None:
                    raise ValueError("Search result is None")
                search_results = search_result.get("result", [])
                for video in search_results:
                    duration_str = video.get("duration")
                    if duration_str and ":" in duration_str:
                        duration_parts = duration_str.split(":")
                        duration_seconds = sum(
                            int(x) * 60**i
                            for i, x in enumerate(reversed(duration_parts))
                        )
                        if duration_seconds >= min_duration:
                            results.append((video["link"], duration_seconds))
                if not search.next():
                    break
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                rprint(
                    f"[bold red]Failed to search for '{query}' after {max_retries} attempts: {str(e)}"
                )
                return []


def deduplicate(file_path: str) -> None:
    rprint(f"[yellow]Deduplicating URLs in {file_path}")
    with open(file_path, "r") as f:
        urls = f.readlines()

    original_count = len(urls)
    deduplicated_urls = list(dict.fromkeys(urls))
    new_count = len(deduplicated_urls)

    with open(file_path, "w") as f:
        f.writelines(deduplicated_urls)

    rprint(
        f"[green]Deduplication complete. Removed {original_count - new_count} duplicate URLs."
    )


def format_duration(seconds: int) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def index_youtube_urls(config: IndexConfig) -> None:
    rprint("[cyan]Starting YouTube URL indexing process")

    deduplicate(config.output_file)

    # Read existing URLs and deduplicate
    existing_urls = set()
    if os.path.exists(config.output_file):
        rprint(f"[yellow]Reading existing URLs from {config.output_file}...")
        with open(config.output_file, "r") as f:
            existing_urls = set(line.strip() for line in f)

    rprint(f"[yellow]Reading search terms from {config.input_file}...")
    with open(config.input_file, "r") as f:
        search_terms = f.read().splitlines()
    rprint(f"[green]Found {len(search_terms)} search terms.")

    # Prepare the search function with configured parameters
    def configured_search(term: str) -> list[tuple[str, int]]:
        return search_youtube(
            term,
            config.min_duration,
            config.max_retries,
            config.search_limit,
            config.retry_delay,
        )

    rprint("[cyan]Beginning search process")
    # Open the output file in append mode
    with open(config.output_file, "a") as out_file:
        total_duration = 0
        pbar = tqdm(
            total=len(search_terms), desc="Searching", disable=not config.progress_bar
        )

        if config.use_multithreading:
            rprint(
                f"[yellow]Using multithreaded search with {config.max_workers} workers."
            )
            # Use ThreadPoolExecutor for multithreading
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_workers
            ) as executor:
                # Submit all search tasks
                futures = executor.map(configured_search, search_terms)
                # Use tqdm for progress bar
                for results in futures:
                    for url, duration in results:
                        if url not in existing_urls:
                            out_file.write(f"{url}\n")
                            existing_urls.add(url)
                            total_duration += duration
                    pbar.set_postfix(
                        {"Total Duration": format_duration(total_duration)},
                        refresh=True,
                    )
                    pbar.update(1)
                    time.sleep(
                        config.multithreaded_sleep
                    )  # Add sleep to avoid rate limits
        else:
            rprint("[yellow]Using single-threaded search.")
            # Single-threaded execution with tqdm progress bar
            for term in search_terms:
                results = configured_search(term)
                for url, duration in results:
                    if url not in existing_urls:
                        out_file.write(f"{url}\n")
                        existing_urls.add(url)
                        total_duration += duration
                pbar.set_postfix(
                    {"Total Duration": format_duration(total_duration)}, refresh=True
                )
                pbar.update(1)

        pbar.close()

    rprint("[cyan]Search process complete. Performing final deduplication...")
    # Final deduplication
    deduplicate(config.output_file)

    rprint("[green]Indexing process complete")
    rprint(f"[cyan]Total unique videos found: {len(existing_urls)}")
    rprint(f"[cyan]Total duration scraped: {format_duration(total_duration)}")


if __name__ == "__main__":
    index_youtube_urls(IndexConfig())
