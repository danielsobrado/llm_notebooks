# LLMs

Large Language Models (LLMs) are created to produce writing that resembles that of a person. They get the ability to anticipate the next word in a sentence based on the context of the previous words through training on a massive amount of text data. They are frequently referred to as autoregressive mathematical models for this reason.

Let's start with some concepts:

**Embedding** is just a vector that represents the significance of a word token or a portion of an image. 

**Context** refers to the model's finite window because it can only handle a small portion of text.

A simple plug-in model called **LoRA** for adjusting the main model's "loss".

**loss** is a score that indicates how poor the output was.

Making a low precision version of the model using **quantization** such that it still functions but is now considerably faster and requires less computing.

When you feed its most recent prediction back in, **GPT** can generate long sentences because it is trained on everything and predicts the next word.

Note: I tested these steps on WSL2 - Ubuntu 20.04.

## About formats and conversions

Language models can be saved and loaded in various formats, here are the most known frameworks:

**PyTorch Model (.pt or .pth)**: This is a common format for models trained using the PyTorch framework. It represents the state_dict (or the "state dictionary"), which is a Python dictionary object that maps each layer in the model to its trainable parameters (weights and biases).

**TensorFlow Checkpoints**: TensorFlow is another popular framework for training machine learning models. A checkpoint file contains the weights of a trained model. Unlike a full model, it doesn't contain any description of the computation that the model performs, it's just the weights. That's useful because often the models can be large and storing the full model in memory can be expensive.

**Hugging Face Transformers**: Hugging Face is a company known for their Transformers library, which provides state-of-the-art general-purpose architectures. They have their own model saving and loading mechanisms, usually leveraging PyTorch or TensorFlow under the hood. You can save a model using the save_pretrained() method and load it using from_pretrained().

Here is a brief overview of the different language model file formats:

* **GGML** stands for Google's Transformer-XL model format. It is a text-based format that stores the model's parameters in a human-readable format. GGML is a good choice for debugging and understanding how the model works.
* **HF** stands for Hugging Face's Transformers format. It is a binary format that stores the model's parameters in a compressed format. HF is a good choice for production deployment, as it is more efficient than GGML.
* **Checkpoints .ckpt** are saved states of a language model's training process. They can be used to resume training, or to load a model that has already been trained. Checkpoints can be useful for debugging, or for saving a model's progress so that it can be resumed later.
* **ONNX** is a cross-platform format for machine learning models. It can be used to store and share language models between different frameworks.
* **Safetensor** is a new format for storing tensors safely. It is designed to be more secure than traditional formats, such as pickle, which can be used to execute arbitrary code. Safetensor is also faster than pickle, making it a good choice for production deployment.
* **Pytorch .pb** is a binary format for storing neural networks. It is efficient and can be loaded quickly.
* **Pytorch .pt** is the most common extension for PyTorch language models. It is a binary file that stores the model's parameters and state.
* **Pytorch .pth** is another common extension for PyTorch language models. It is a text-based file that stores the model's parameters and state.
* **.bin** file is a binary file that stores the parameters and state of a language model. It is a more efficient way to store a language model than a text-based file, such as a .pth file. This is because a binary file can be compressed, which makes it smaller and faster to load.

### Quantization 

Quantization is a technique for reducing the size and complexity of machine learning models. It works by representing the model's parameters and weights in a lower precision format. This can lead to significant reductions in model size and inference time, without sacrificing much accuracy.

There are two main types of quantization: post-training quantization and quantization aware training.

**Post-training quantization** is the most common type of quantization. It works by converting a trained model to a lower precision format after it has been trained. This can be done using a variety of tools and techniques.

**Quantization aware training** is a newer technique that involves training a model with quantization in mind. This can lead to better accuracy and performance than post-training quantization.

### Convert Open-LLama Checkpoint to quantized GGML format

Download Open LLama:
```
git clone https://huggingface.co/openlm-research/open_llama_7b_preview_200bt/
```

Clone llama.cpp and build it:
```
git clone https://github.com/ggerganov/llama.cpp 
cd llama.cpp
cmake -B build 
cmake --build build
```

Convert it from ```.pth``` to ```.ggml```:

```
python3 convert-pth-to-ggml.py models/open_llama_7b_preview_200bt/open_llama_7b_preview_200bt_transformers_weights 1
```

## LLM notebooks
Testing local LLMs 

1. Train Vicuna 7B on a text fragment: [Notebook](https://github.com/danielsobrado/llm_notebooks/blob/main/Notebooks/Train%20Vicuna%207b.ipynb)

## Links:
* GPTQ inference Triton kernel: https://github.com/fpgaminer/GPTQ-triton
