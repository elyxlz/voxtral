import gradio as gr
import torch
import torchaudio
from pydantic_settings import BaseSettings
from voxtral.tokenizer.model import VoxtralTokenizer, VoxtralTokenizerConfig
import transformers as tr


class ServerConfig(BaseSettings):
    share: bool = True
    # voxtral_path: str = "elyx/voxtral-overfit"
    voxtral_path: str = "nilq/mistral-1L-tiny"
    voxtral_revision: str | None = None
    voxtral_tokenizer_config: VoxtralTokenizerConfig = VoxtralTokenizerConfig()


def process_audio(
    model,
    tokenizer,
    audio,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):
    # Resample the audio to 24kHz
    waveform, sample_rate = torchaudio.load(audio)
    if sample_rate != 24000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 24000)

    # Ensure the audio is mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Tokenize the audio
    tokens = tokenizer.encode(waveform.unsqueeze(0), 24000)

    # Generate continuation
    with torch.no_grad():
        output_tokens = model.generate(
            tokens.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

    # Decode the tokens back to audio
    output_audio = tokenizer.decode(output_tokens.squeeze(0))

    # Save the output audio
    output_path = "output.wav"
    torchaudio.save(output_path, output_audio, 24000)

    return output_path


def serve(config: ServerConfig):
    # Load the Voxtral model
    model = tr.MistralForCausalLM.from_pretrained(
        config.voxtral_path,
        revision=config.voxtral_revision,
    )
    model.eval()

    # Initialize the tokenizer
    tokenizer = VoxtralTokenizer(config.voxtral_tokenizer_config)

    # Create the Gradio interface
    iface = gr.Interface(
        fn=lambda *args: process_audio(model, tokenizer, *args),
        inputs=[
            gr.Audio(type="filepath", label="Input Audio"),
            gr.Slider(
                minimum=1, maximum=1000, value=100, step=1, label="Max New Tokens"
            ),
            gr.Slider(
                minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
            ),
            gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k"),
            gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-p"),
            gr.Slider(
                minimum=1.0,
                maximum=2.0,
                value=1.2,
                step=0.1,
                label="Repetition Penalty",
            ),
        ],
        outputs=gr.Audio(type="filepath", label="Generated Audio"),
        title="Voxtral Audio Continuation",
        description="Upload an audio file to continue it using Voxtral.",
    )

    # Launch the interface
    iface.launch(share=config.share)
