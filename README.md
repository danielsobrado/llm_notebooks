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

### Models available

**Llama**

Llama is a large language model (LLM) released by Meta AI in February 2023. A variety of model sizes were trained ranging from 7 billion to 65 billion parameters. LLaMA's developers reported that the 13 billion parameter model's performance on most NLP benchmarks exceeded that of the much larger GPT-3 (with 175 billion parameters) and that the largest model was competitive with state of the art models such as PaLM and Chinchilla.

**Open Llama**

[Open Llama](https://github.com/s-JoL/Open-Llama) is an open-source reproduction of Meta AI's LLaMA model. The creators of Open Llama have made the permissively licensed model publicly available as a 7B OpenLLaMA model that has been trained with 200 billion tokens.

**Vicuna**
Vicuna is a delta model for LLaMA. Delta models are small, efficient models that can be used to improve the performance of larger models. Vicuna Delta is trained on a dataset of user-shared conversations collected from ShareGPT, and it has been shown to improve the performance of LLaMA on a variety of NLP tasks, including natural language inference, question answering, and summarization.

Vicuna Delta is available as a pre-trained model from the Hugging Face Hub.

## Convert Open-LLama Checkpoint to quantized GGML format

Download Open LLama into your models folder:
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
python3 convert-pth-to-ggml.py ../models/open_llama_7b_preview_200bt/ open_llama_7b_preview_200bt_transformers_weights 1
```

![convertToGGML](https://github.com/danielsobrado/llm_notebooks/blob/main/images/convertToGGML.png)

Test it:

./build/bin/main -m models/open_llama_7b_preview_200bt_q5_0.ggml --ignore-eos -n 1280 -p "Give me in python the quicksort algorithm" --mlock

![executeGGMLF16](https://github.com/danielsobrado/llm_notebooks/blob/main/images/executeGGMLF16.png)

Quantize it to 4 bits:

```
./build/bin/quantize ../models/open_llama_7b_preview_200bt/open_llama_7b_preview_200bt_transformers_weights/ggml-model-f16.bin ../models/open_llama_7b_preview_200bt_q4_0.ggml q4_0
```
![quantizeTo4Bits](https://github.com/danielsobrado/llm_notebooks/blob/main/images/quantizeTo4Bits.png)

It is way smaller!

![quantizedModel](https://github.com/danielsobrado/llm_notebooks/blob/main/images/quantizedModel.png)

Test it:

./build/bin/main -m models/open_llama_7b_preview_200bt_q4_0.ggml --ignore-eos -n 1280 -p "Give me in python the quicksort algorithm" --mlock

![executeGGML4B](https://github.com/danielsobrado/llm_notebooks/blob/main/images/executeGGML4B.png)

You'll notice that the inference is much faster and requires less memory.

## LLM notebooks
Testing local LLMs 

1. Train Vicuna 7B on a text fragment: [Notebook](https://github.com/danielsobrado/llm_notebooks/blob/main/Notebooks/Train%20Vicuna%207b.ipynb)

## Concepts:
1. Dot Product: [Notebook](https://github.com/danielsobrado/llm_notebooks/blob/main/Notebooks/Dot%20Product.ipynb)

## How to run the examples

Folow the steps:

* Create a new environment: `conda create -n examples`
* Activate the environment: `conda activate examples`
* Install the packages: `conda install jupyter numpy matplotlib seaborn plotly`
* Start notebooks: `jupyter notebook --NotebookApp.password="''"  --NotebookApp.token="''"`
 
## Links:
* GPTQ inference Triton kernel: https://github.com/fpgaminer/GPTQ-triton

# LLM Server

We'll use [oobabooga](https://github.com/oobabooga/text-generation-webui) as server, using the [OpenAI extension](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)

* `python server.py --extensions openai --no-stream`

We'll download a model GGML for CPU or GPTQ for GPU, both quantized:
![oobabooga1](https://github.com/danielsobrado/llm_notebooks/blob/main/images/LLM_oobabooga.png)
![oobabooga2](https://github.com/danielsobrado/llm_notebooks/blob/main/images/LLM_oobabooga2.png)

Test that works fine:
![oobabooga3](https://github.com/danielsobrado/llm_notebooks/blob/main/images/LLM_oobabooga3.png)

And we'll be able to connect from our notebook to the server:
![oobabooga server](https://github.com/danielsobrado/llm_notebooks/blob/main/images/LLM_oobabooga4.png)




