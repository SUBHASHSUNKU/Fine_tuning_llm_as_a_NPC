# Fine_tuning_llm_as_a_NPC
Built an AI-powered Martian NPC that learns from player–character conversations and generates interactive responses.
The project simulates how non-player characters can engage in natural dialogue within games.
# Fine-Tuning Gemma 3 (270M) with Unsloth

This project demonstrates how to fine-tune Google's Gemma 3 270M Instruct model using LoRA (PEFT) with the Unsloth framework. The model is trained on a conversational NPC dataset and can generate responses in a chat format.

## Features

* Loads Gemma 3 270M Instruct model in 4-bit quantization
* Uses LoRA (Low-Rank Adaptation) for efficient fine-tuning
* Supports memory-efficient training with Unsloth
* Chat template formatting for conversational datasets
* Fast inference before and after training
* Suitable for Google Colab and consumer GPUs

## Tech Stack

* Python
* Unsloth
* Transformers
* TRL
* PEFT
* Datasets
* BitsAndBytes

## Model

Base Model:

`unsloth/gemma-3-270m-it`

## Dataset

Dataset Used:

`bebechien/MobileGameNPC`

Configuration:

`martian`

The dataset contains conversational interactions between a player and an NPC character.

## Installation

```bash
pip install -U --no-cache-dir unsloth unsloth_zoo
pip install -U --no-deps trl peft accelerate bitsandbytes xformers
```

## Training Configuration

| Parameter             | Value       |
| --------------------- | ----------- |
| LoRA Rank (r)         | 8           |
| LoRA Alpha            | 8           |
| Batch Size            | 2           |
| Gradient Accumulation | 4           |
| Epochs                | 30          |
| Learning Rate         | 2e-4        |
| Optimizer             | AdamW 8-bit |
| Max Sequence Length   | 2048        |
| Quantization          | 4-bit       |

## Workflow

1. Load pre-trained Gemma model
2. Run baseline inference
3. Apply LoRA adapters using PEFT
4. Load and preprocess dataset
5. Format conversations using Gemma chat template
6. Fine-tune using SFTTrainer
7. Switch model to inference mode
8. Generate responses with the fine-tuned model

## Example Inference

```python
messages = [
    {
        "role": "user",
        "content": "Hello there."
    }
]

do_inference(messages)
```

## Project Structure

```text
.
├── finetuning_llm.py
├── output/
└── README.md
```

## Notes

* Designed for educational and experimentation purposes.
* Training checkpoints are saved in the `output/` directory.
* GPU acceleration is required for practical training times.
* Works particularly well on Google Colab environments.

## Future Improvements

* Support custom datasets
* Experiment with larger Gemma variants
* Add evaluation metrics
* Compare LoRA ranks and hyperparameters
* Export and publish adapters to Hugging Face

## License

This project follows the licenses of the respective libraries, models, and datasets used.
