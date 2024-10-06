# Voxtral

<p align="center"><img src="./assets/voxtral.png" alt="Voxtral" width="300"></p>

My submission for the A16z x Cerebral Valley Mistral London Hackathon

## About

This repo is a full stack pipeline to convert a mistral model into an end2end SpeechLM (like GPT4o voice mode or Moshi).

SpeechLMs (audio in and audio out) are the missing ingredient to truly human like conversational agents.

The main reason is because they don't have the information bottleneck typical speech systems have (audio -> text -> audio) where all prosody, rhythm, tone, and emotional information is lost in the ASR process and has to hallucinated in the TTS stage. Furthermore, speechLMs can learn how to handle interruptions and silence directly from the data without special and unreliable hacks.

Only two working speechLMs have every been produced publicly: GPT4o and Moshi. Gpt4o is closed source and has been safety tuned and synthetically post-trained to oblivion. Moshi IMO has too complicated of an architecture (for very little gain) and has also been trained with too much synthetic data, that leaves it still feeling robotic (defeating the entire point of SpeechLMs!)

This repo provides the infrastructure for anyone with modest gpu capacity to create their own successful speechLM.

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
