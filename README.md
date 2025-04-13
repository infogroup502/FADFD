# Frequency-domain Spectrum Discrepancy-based Fast Anomaly Detection for IIoT Sensor Time-Series Signals (TIM 2025)
This repository provides a PyTorch implementation of PPLAD ([paper](https://ieeexplore.ieee.org/abstract/document/10938317)).

## Framework
<img src="https://github.com/infogroup502/FADFD/blob/main/img/workflow.png" width="850px">

## Main Result
<img src="https://github.com/infogroup502/FADFD/blob/main/img/result.png" width="850px">

## Requirements
The recommended requirements for FADFD are specified as follows:
-arch==6.1.0
-einops==0.6.1
-matplotlib==3.7.0
-numpy==1.23.5
-pandas==1.5.3
-Pillow==9.4.0
-scikit_learn==1.2.2
-scipy==1.8.1
-statsmodels==0.14.0
-torch==1.13.0
-tqdm==4.65.0
-tsfresh==0.20.1

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data
The datasets can be obtained and put into datasets/ folder in the following way:
- Our model supports anomaly detection for multivariate time series datasets.
- We provide the SKAB dataset. If you want to use your own dataset, please place your datasetfiles in the `/dataset/<dataset>/` folder, following the format `<dataset>_train.npy`, `<dataset>_test.npy`, `<dataset>_test_label.npy`.

## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- main.py: The main python file. You can adjustment all parameters in there.
- metrics: There is the evaluation metrics code folder.
- model: FADFD model folder
- solver.py: Another python file. The training, validation, and testing processing are all in there
- requirements.txt: Python packages needed to run this repo


- ## Usage
1. Install Python 3.9, PyTorch >= 1.4.0
2. Download the datasets
3. To train and evaluate FADFD on a dataset, run the following command:
```bash
python main.py 
```
## BibTex Citation
```bash
@ARTICLE{10938317,
  author={Chen, Lei and Liu, Xuxin and Zou, Ying and Tang, Jiajun and Liu, Canwei and Hu, Bowen and Lv, Mingyang},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Frequency-Domain Spectrum Discrepancy-Based Fast Anomaly Detection for IIoT Sensor Time-Series Signals}, 
  year={2025},
  volume={74},
  number={},
  pages={1-16},
  keywords={Anomaly detection;Industrial Internet of Things;Computational modeling;Accuracy;Time-frequency analysis;Feature extraction;Image edge detection;Neural networks;Cloud computing;Transformers;Anomaly detection;fast anomaly detection;frequency domain;Industrial Internet of Things (IIoT);sensor signal;spectrum discrepancy},
  doi={10.1109/TIM.2025.3554286}
  }
```
