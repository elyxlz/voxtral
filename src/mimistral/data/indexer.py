import concurrent.futures
from youtubesearchpython import VideosSearch
from tqdm import tqdm
import time


def search_youtube(query, min_duration=30 * 60, max_retries=3):
    for attempt in range(max_retries):
        try:
            search = VideosSearch(query, limit=100)
            results = []
            while len(results) < 100:
                search_results = search.result()["result"]
                for video in search_results:
                    duration_str = video["duration"]
                    if ":" in duration_str:
                        duration_parts = duration_str.split(":")
                        duration_seconds = sum(
                            int(x) * 60**i
                            for i, x in enumerate(reversed(duration_parts))
                        )
                        if duration_seconds >= min_duration:
                            results.append(video["link"])
                if not search.next():
                    break
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait for 2 seconds before retrying
            else:
                print(
                    f"Failed to search for '{query}' after {max_retries} attempts: {str(e)}"
                )
                return []


def search_wrapper(term):
    return search_youtube(term)


def main():
    with open("searches.txt", "r") as f:
        search_terms = f.read().splitlines()

    all_results = []

    # Use ThreadPoolExecutor for multithreading
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all search tasks and wrap with tqdm for progress bar
        futures = list(
            tqdm(
                executor.map(search_wrapper, search_terms),
                total=len(search_terms),
                desc="Searching",
            )
        )
        # Collect results
        for future in futures:
            all_results.extend(future)

    # Deduplicate results
    deduplicated_results = list(dict.fromkeys(all_results))

    with open("urls.txt", "w") as f:
        for url in deduplicated_results:
            f.write(f"{url}\n")

    print(f"Total unique videos found: {len(deduplicated_results)}")


if __name__ == "__main__":
    main()
