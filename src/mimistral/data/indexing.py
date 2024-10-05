import concurrent.futures
from youtubesearchpython import VideosSearch
from tqdm import tqdm
import time
import typing


class IndexConfiguration(typing.NamedTuple):
    input_file: str = "./data/searches.txt"
    output_file: str = "./data/urls.txt"
    min_duration: int = 30 * 60
    max_retries: int = 3
    search_limit: int = 10
    retry_delay: float = 2.0
    max_workers: int = 1
    progress_bar: bool = True
    use_multithreading: bool = False


def search_youtube(
    query: str,
    min_duration: int,
    max_retries: int,
    search_limit: int,
    retry_delay: float,
) -> list[str]:
    for attempt in range(max_retries):
        try:
            search = VideosSearch(query, limit=search_limit)
            results: list[str] = []
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
                            results.append(video["link"])
                if not search.next():
                    break
            print("yo")
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(
                    f"Failed to search for '{query}' after {max_retries} attempts: {str(e)}"
                )
                return []


def index_youtube_urls(config: IndexConfiguration) -> None:
    with open(config.input_file, "r") as f:
        search_terms = f.read().splitlines()

    all_results: list[str] = []

    # Prepare the search function with configured parameters
    configured_search = lambda term: search_youtube(
        term,
        config.min_duration,
        config.max_retries,
        config.search_limit,
        config.retry_delay,
    )

    if config.use_multithreading:
        # Use ThreadPoolExecutor for multithreading
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers
        ) as executor:
            # Submit all search tasks
            futures = executor.map(configured_search, search_terms)

            # Wrap with tqdm for progress bar if enabled
            if config.progress_bar:
                futures = tqdm(futures, total=len(search_terms), desc="Searching")

            # Collect results
            for future in futures:
                all_results.extend(future)
    else:
        # Single-threaded execution
        for term in tqdm(
            search_terms, desc="Searching", disable=not config.progress_bar
        ):
            results = configured_search(term)
            all_results.extend(results)

    # Deduplicate results
    deduplicated_results = list(dict.fromkeys(all_results))

    with open(config.output_file, "w") as f:
        for url in deduplicated_results:
            f.write(f"{url}\n")

    print(f"Total unique videos found: {len(deduplicated_results)}")


if __name__ == "__main__":
    index_youtube_urls(IndexConfiguration())
