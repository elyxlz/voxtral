from voxtral.data.indexing import IndexConfig, index_youtube_urls

config = IndexConfig()
index_youtube_urls(config)


from voxtral.data.scraping import ScrapingConfig, scrape_youtube_urls

config = ScrapingConfig()
scrape_youtube_urls(config)


from voxtral.data.preprocessing import PreprocessingConfig, preprocess_audio_chunks

config = PreprocessingConfig()
preprocess_audio_chunks(config)
