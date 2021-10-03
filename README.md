# Self-Attention based Fine-Grained Cross-Media Hybrid Network
## Introduction
This is the Pytorch implementation for our paper: **Self-Attention based Fine-Grained Cross-Media Hybrid Network**
## Network Architecture
The architecture of our proposed approach is as follows
<div align=center><img  src="https://github.com/NUST-Machine-Intelligence-Laboratory/SAFGCMHN/blob/main/fig/architecture.png"/></div>

## Installation
**Environment**  
* pytorch, tested on [v1.8.1]  
* CUDA, tested on v10.0  
* Language: Python 3.7

Create a virtual environment with python 3.7,

    $  conda create -n safg_env python=3.7

    $  conda activate safg_env

  Install all dependencies

    $  pip install -r requirements.txt
    
## Download
* **Download dataset**  
Please visit this [project page](http://59.108.48.34/tiki/FGCrossNet/).
* **Pretrained model**
Download the pretrained [SAN](https://github.com/hszhao/SAN) model.

## Training
   * If you want to train the whole model from beginning, please Activate virtual environment (e.g. conda) and then run the script.
      ```python
      $  python main.py
      ```
   * Or directly utilize our trained model from [Baidu Cloud](https://pan.baidu.com/s/1GlDbEbZizk5jncEwXlbpig) with a password `fwb8`.
## Testing 
   * Run `test.py` to extract image, audio and text features.
      ```python
      $  python test.py
      ```
    * Extract video features.
      ```python
      $  python video_feature.py
      $  python video_feature_cal.py
      ```
## Results
 &emsp;&emsp;The final experimental results are shown in the following table：    
 <div align=center><img  src="https://github.com/NUST-Machine-Intelligence-Laboratory/SAFGCMHN/blob/main/fig/table2.png"/></div>
 <div align=center><img  src="https://github.com/NUST-Machine-Intelligence-Laboratory/SAFGCMHN/blob/main/fig/table3.png"/></div>
