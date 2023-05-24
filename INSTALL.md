# 🛠️ Installation

## Basic requirements

- Linux 
- Python 3.8+ 
- PyTorch 1.12+
- CUDA 11.6+ 
- GCC & G++ 5.4+
- GPU Memory >= 17G for loading basic tools (HuskyVQA, SegmentAnything, ImageOCRRecognition)


## Install Python dependencies

```shell
conda create -n ichat python=3.8
conda activate ichat
pip install -r requirements.txt
```


## 🗃 Model zoo

our `model_zoo` has been released in [huggingface](https://huggingface.co/spaces/OpenGVLab/InternGPT/tree/main/model_zoo)! 
You can download it and directly place it into the root directory of this repo before running the app.

HuskyVQA, a strong VQA model, is also available in `model_zoo`. More details can refer to our [report](https://arxiv.org/pdf/2305.05662.pdf).

**Note for Husky checkpoint**

Due to the license issuse, we could directly provide the checkpoint of Husky. The `model_zoo` contains the delta checkpoint between Husky and [LLAMA](https://github.com/facebookresearch/llama). 

To build the actual checkpoint of Husky, you need the original checkpoint of LLAMA, which should be put in `model_zoo/llama/7B`. We support automatically download the llama checkpoint, but you need to request a form for the download url from Meta (see [here]((https://github.com/facebookresearch/llama))). Once you have the download url, paste it into `PRESIGNED_URL=""` at [third-party/llama_download.sh](third-party/llama_download.sh).

Then, rerun the app would automatically download the original checkpoint, convert it to huggingface format, and build the Husky checkpoint. 

Please make sure these folder `model_zoo/llama`, and  `model_zoo/llama_7B_hf` contain the correct checkpoint, otherwise you should delete the folder and let the app download it again.
Otherwise, you might encounter issuses similar as [issue #5](https://github.com/OpenGVLab/InternGPT/issues/5)
> model_zoo\llama_7B_hf does not appear to have a file named config.json. 

> FileNotFoundError: [Errno 2] No such file or directory: 'model_zoo/llama/7B/params.json'