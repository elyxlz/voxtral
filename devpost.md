# Full Stack Pipeline for SpeechLM

This repo is a full stack pipeline to convert a Mistral model into an end-to-end SpeechLM (like GPT4o voice mode or Moshi).

## Why SpeechLMs?

SpeechLMs (audio in and audio out) are the missing ingredient for truly human-like conversational agents. The main reason is they don't have the information bottleneck typical speech systems have (audio -> text -> audio) where all prosody, rhythm, tone, and emotional information is lost in the ASR process and has to be hallucinated in the TTS stage. Furthermore, SpeechLMs can learn how to handle interruptions and silence directly from the data without special and unreliable hacks.

Only two working SpeechLMs have ever been produced publicly: GPT4o and Moshi. GPT4o is closed source and has been safety tuned and synthetically post-trained to oblivion. Moshi IMO has too complicated of an architecture (for very little gain) and has also been trained with too much synthetic data, that leaves it still feeling robotic (defeating the entire point of SpeechLMs!)

This repo provides the infrastructure for anyone with modest GPU capacity to create their own successful SpeechLM.

## Architecture

For the architecture, I decided to go against the typical approach of making multimodal LLMs which consists of getting continuous embeddings from some encoder and sticking them in the sequence. That's because that would only allow for the extra modality to be used as an input, not as an output (not without an ugly arch anyways). So I decided to make everything discrete and just extend the vocabulary of Mistral. This simplifies things as I could just use a Mistral base model out the bat without changing anything except make a new tokenizer.

Key points:

- Used Mimi (best opensource neural audio codec for speech) for audio tokenization
- 4 quantizers (50Hz) to speed up training, although this does harm audio quality
- Interleaved text tokens using time-stamped Whisper (a modification of Whisper that uses the cross attention weights to timestamp individual tokens)
- Created a 55Hz multimodal audio tokenizer for encoding data and finetuning Mistral
- Added extra delay between text and audio tokens to ensure correct ordering

## Pipeline Stages

1. Indexing
2. Scraping
3. Preprocessing
4. Training
5. Serving

### Indexing

To train a good SpeechLM, you need good data. This is IMO where GPT4o and Moshi messed up - they used too much synthetic or low-quality data which makes their SpeechLMs still sound robotic (albeit way less than typical voice systems). This means no audiobooks or things of the sort. I wanted podcasts.

Unfortunately, podcast datasets don't really exist on Hugging Face or anywhere else. So I had to implement my own podcast dataset pipeline in this repo. That starts with indexing a lot of podcasts on YouTube. I had Claude generate a list of 500 search terms that would result in podcasts and then made a script using an unofficial YouTube search API to index them all in a multithreaded way.

### Scraping

Next would be downloading and chunking the YouTube videos. For this I used yt-dlp and ffmpeg. The Nebius machines had good bandwidth so I could pump up the num workers to a large amount ~128. I made sure not to transcode the files or I would incur a heavy slowdown. I found that ffmpeg has a `segment` option that allows for chunking of audio files without transcoding. I used this and got a great scraping realtime factor of about 8000x. I downloaded about 8k hours of high-quality conversational speech data in an hour.

### Preprocessing

For efficient training, I would have to pretokenize all my data. So I use a PyTorch DataLoader to efficiently load all of my audio chunks into tensors, pad them and rechannel to the correct amount, and encode them in batches on the GPU. I saved all of the encoded tokens into individual and hashed npy files for training.

### Training

To train I would need to implement an efficient Mistral training script that fit on a single GPU. I used the transformers Mistral model definition to save time but implemented my own trainer because I've had bad experiences with the Hugging Face trainers before (too high level for me).

Since we are training on a new modality, LoRA would probably not be enough, so another way of getting a 7B model to fit on a single GPU was to prune its weights. I went for a pretty stupid strategy of just dropping out every other layer. I think I've seen this done somewhere. I expose both this option and the option to use LoRA.

Training is pretty standard other than this:

- Added some W&B logging where it would sample from the model given a 10s audio prompt
- TODO: Add ability to weight the loss on different types of tokens (weigh text tokens higher than semantic tokens, and acoustic tokens the least)

Training loads all the numpy files in the directory.

### Serving

Although I didn't have time to train a great model (would have probably been impossible with a single GPU as well), I made a little Gradio UI where you can speak with the microphone to prompt the model and get a response back.

Future work would be to have a constant streaming conversation, which would be doable given Mistral's fantastic architecture with sliding window attention and a rolling KV cache.

For this demo, I served a version of the model which overfit on a few audio samples (to prove it works).

## Notes

- Current demo uses an overfitted model on few audio samples (proof of concept)
- Single GPU training limits model quality
- Future improvements: loss weighting, streaming conversations
