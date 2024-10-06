# Voxtral

<p align="center"><img src="./assets/voxtral.png" alt="Voxtral" width="300"></p>

My submission for the A16z x Cerebral Valley Mistral London Hackathon

## About

Voxtral: Convert Mistral into a end2end SpeechLM. No information bottleneck, preserves prosody, learns interruptions from data. Unlike GPT4o (closed) or Moshi (complex), it's open, simple, natural.

## Tech Stack

- PyTorch

## Usage

This repo uses `uv`, the best python package manager.
Install like this:

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

This repo has 5 stages:

1. **Indexing**: This is where youtube urls of podcasts or other conversational audios are obtained via an unofficial youtube search api. The search terms were generated with Claude, an example of 500 terms is in `./data/searches.txt`

2. **Scraping**: This is where the youtube urls are downloaded and chunked with `yt-dlp` and `ffmpeg` in an optimized mulitthreaded way. With good internet connection expect to download data at a 5000 - 10000x realtime rate (10k hours in 1 hour)

3. **Preprocessing**: This is where the speech audio chunks are efficient encoded on gpus using the custom multimodal tokenizer (more on that later)

4. **Training**: This is where mistral is finetuned from pretrained weights on the new multimodal tokens

5. **Serving**: A simple gradio ui is loaded to interface with the model via voice.

To run all the phases sequentially with the default configuration (you should be on a machine with at least an A100 worth of compute), run:

```sh
uv run ./scripts/everything.py
```

Alternatively, you can run each stage independently and alter the config either by passing environment variable flags or by changing the script:

- index.py
- scrape.py
- preprocess.py
- train.py
- serve.py

## LICENSE

MIT
