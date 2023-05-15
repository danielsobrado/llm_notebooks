# LLMs

## About formats and conversions

### Convert Open LLama to GGML format

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

## LLM notebooks
Testing local LLMs 

1. Train Vicuna 7B on a text fragment: [Notebook](https://github.com/danielsobrado/llm_notebooks/blob/main/Notebooks/Train%20Vicuna%207b.ipynb)

## Links:
* GPTQ inference Triton kernel: https://github.com/fpgaminer/GPTQ-triton
