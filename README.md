# Introduction
Recurrent Dynamic Embedding for Video Object Segmentation [CVPR 2022]

# Install
If you just implement our method, refer to [Requirements](https://github.com/hkchengrex/STCN#requirements).  
If you want to evaluate our method on the davis 2017 validation set, refer to [Requirements](https://github.com/davisvideochallenge/davis2017-evaluation#installation).

# Model zoo
You can download the pretrained models from [Google](https://drive.google.com/drive/folders/1i6GtLPUNZe0h72IfhU2s_WHA7ZuVoudG?usp=sharing).

The predictions of our method can be download from [Google](https://drive.google.com/drive/folders/1cQLvJwoNWV__kiWG_hiTcBZa1eXrWoqm?usp=sharing).


# Dataset
Following [STCN](https://arxiv.org/pdf/2106.05210), we train the network in the three stages. Firstly, we train the network on the static image dataset, which can be downloaded in `download_datasets.py`. Then we fine-tune the network with SAM on the BL30K dataset, which can be downloaded in `download_bl30k.py`. Note, BL30K is an extensive dataset introduced by [MiVOS](https://arxiv.org/pdf/2103.07941.pdf) and is 700GB in total. Finally, we fine-tune the network with SAM on the mixed dataset (DAVIS 2017 and YouTube-VOS 2019).  

I know it doesn't look straightforward., **while you can just download DAVIS 2017 and have a quick start right away**.

```bash
├── BL30K
├── DAVIS
│   ├── 2016
│   │   ├── Annotations
│   │   └── ...
│   └── 2017
│       ├── test-dev
│       │   ├── Annotations
│       │   └── ...
│       └── trainval
│           ├── Annotations
│           └── ...
├── static
│   ├── BIG_small
│   └── ...
├── YouTube
│   ├── all_frames
│   │   └── valid_all_frames
│   ├── train
│   ├── train_480p
│   └── valid
└── YouTube2018
    ├── all_frames
    │   └── valid_all_frames
    └── valid
```

# Quick start 
Take the inference on the Davis 2017 validation set as an example. The inference command is as follows:
```python
python eval_davis.py --output ... --davis_path ... --model  ...   --mode two-frames-compress  --mem_every ... --top ... --amp
```

- output: prediction path to save the results.
- davis_path: the path for Davis 2017.
- model: pre-trained model path.
- mode: optional items, such as `two-frames-compress`, `gt-compress`, `last-compress`.
- mem_every: the interval to use SAM.  
- top: topk-filter.
- amp: use apex to infer.

You can use this protocol.
```python
python eval_davis.py --output prediction/s012 --davis_path ... --model  pretrain/model_s012_final.pth   --mode two-frames-compress  --mem_every 3 --amp
```


# Quick evaluation
Take the evaluation on the Davis 2017 validation set as an example. 

We modify this [repo](https://github.com/davisvideochallenge/davis2017-evaluation) to evaluate our method.
```python
python evaluation/2017/evaluation_ours.py --results_path ... --davis_path ...
```
- results_path: prediction path to save the results.
- davis_path: the path for Davis 2017.


# Results 
## Without BL30K  

| Dataset | Split |  ![](http://latex.codecogs.com/svg.latex?J\\&F) | ![](http://latex.codecogs.com/svg.latex?J) |  ![](http://latex.codecogs.com/svg.latex?F)  | FPS 
| --- | --- | :--:|:--:|:---:|:---:|
| DAVIS 2016 | validation | 91.1 | 89.7 | 92.5 | 35.0
| DAVIS 2017 | validation | 84.2 | 80.8 | 87.5 | 27.0
| DAVIS 2017 | test-dev | 77.4 | 73.6 | 81.2 | -

| Dataset | Split | ![](http://latex.codecogs.com/svg.latex?J\\&F) | ![](http://latex.codecogs.com/svg.latex?{J_{seen}}) | ![](http://latex.codecogs.com/svg.latex?{F_{seen}}) |![](http://latex.codecogs.com/svg.latex?{J_{unseen}}) |![](http://latex.codecogs.com/svg.latex?{F_{unseen}}) |
| --- | --- | :--:|:--:|:---:|:---:|:---:|
| YouTube 2019| validation | 81.9| 81.1|85.5|76.2|84.8|

## With BL30K  

| Dataset | Split | ![](http://latex.codecogs.com/svg.latex?J\\&F) |  ![](http://latex.codecogs.com/svg.latex?J)  |  ![](http://latex.codecogs.com/svg.latex?F)  | FPS 
| --- | --- | :--:|:--:|:---:|:---:|
| DAVIS 2016 | validation | 91.6 | 90.0 | 93.2 | 35.0
| DAVIS 2017 | validation | 86.1 | 82.1 | 90.0 | 27.0
| DAVIS 2017 | test-dev | 78.9 | 74.9 | 92.9 | -

| Dataset | Split | ![](http://latex.codecogs.com/svg.latex?J\\&F) | ![](http://latex.codecogs.com/svg.latex?{J_{seen}}) | ![](http://latex.codecogs.com/svg.latex?{F_{seen}}) |![](http://latex.codecogs.com/svg.latex?{J_{unseen}}) |![](http://latex.codecogs.com/svg.latex?{F_{unseen}}) |
| --- | --- | :--:|:--:|:---:|:---:|:---:|
| YouTube 2019| validation | 83.3| 81.9|86.3|78.0|86.9|

# Inference 
By one gpu, you can infer these datasets as follows:

- **DAVIS 2017 validation set**
```python
python eval_davis.py --output prediction/DAVIS-2017-val --davis_path ... --model  pretrain/model_s012_final.pth   --mode two-frames-compress  --mem_every 3 --amp
```
- **Davis 2017 test set** 
```python
python eval_davis.py --output prediction/DAVIS-2017-test --davis_path ... --model  pretrain/model_s012_final.pth   --mode two-frames-compress  --mem_every 3 --top 40 --split testdev --amp
```
- **DAVIS 2016 validation set**
```python
python eval_davis_2016.py --output prediction/DAVIS-2017-val --davis_path ... --model  pretrain/model_s012_final.pth   --mode two-frames-compress  --mem_every 3 --top 40 --split testdev --amp
```
- **YouTube 2019 validation set**
```python
python eval_youtube.py --output prediction/YV-19-val --yv_path ... --model  pretrain/model_s012_final_yv.pth  --mode two-frames-compress  --mem_every 4 --top 20 --amp
```

# Training
Firstly, you must configure the paths to the dataset in `util/hyper_para.py`,  which include `--static_root`, `--bl_root`, `--yv_root` and `--davis_root`.

## stage 0 
```python
cd rootdir &&\
OMP_NUM_THREADS=4 python -m  torch.distributed.launch --master_port 9843 \
--nproc_per_node=4 \
train.py --id  s0 \
--stage 0 \
--perturb_max 1 \
--perturb_min 0.85 \
--save_interval  10000 \
--klloss_weight 10 \
--start_warm 5000 \
--end_warm 17500 \
--batch_size 16 \
--lr 2e-05 \
--steps 37500 \
--iterations 75000 \
--repeat 0  
```
## stage 0 -> 3 (w/o BL30K)
```python
cd rootdir &&\
OMP_NUM_THREADS=4 python -m  torch.distributed.launch --master_port 9844 \
--nproc_per_node=2 \
train.py --id  s03 \
--stage 3 \
--load_network pretrain/s0/model_75000.pth \
--perturb_max 1 \
--perturb_min 0.85 \
--save_interval  10000 \
--klloss_weight 10 \
--batch_size 4 \
--lr 2e-05 \
--steps 125000 \
--iterations 150000 \
--repeat 0  
```

## stage 1
```python
cd rootdir &&\
OMP_NUM_THREADS=4 python -m  torch.distributed.launch --master_port 9843 \
--nproc_per_node=2 \
train.py --id  s1 \
--stage 1 \
--load_network pretrain/s0/model_75000.pth \
--perturb_max 1 \
--perturb_min 0.85 \
--save_interval  10000 \
--klloss_weight 10 \
--start_warm 20000 \
--end_warm 70000 \
--batch_size 4 \
--lr 1e-05 \
--steps 400000 \
--iterations 500000 \
--repeat 0  
```

## stage 2
```python
cd rootdir &&\
OMP_NUM_THREADS=4 python -m  torch.distributed.launch --master_port 9843 \
--nproc_per_node=2 \
train.py --id  s2 \
--stage 2 \
--load_network /gdata/limx/VOS/SAM/cvpr-22-code/pretrain/s1/model_500000.pth \
--perturb_max 1 \
--perturb_min 0.85 \
--save_interval  10000 \
--klloss_weight 5 \
--decoder_f2_weight 5 \
--decoder_f4_weight 5 \
--start_warm 5000 \
--end_warm 17500 \
--batch_size 8 \
--lr 2e-05 \
--steps 62500 \
--iterations 75000 \
--repeat 0  
```

Note since I suffered temporary layoffs during my internship at Alibaba, there is uncertainty about the installation environment and the version of the code I applied for. I tried to reproduce the previous parameters on this version and got **0.857** on Davis 17 val (0.861 in the original paper) and **79.2** on Davis 17 test (0.789 in the original paper).

The original parameters:
```
klloss_weight = 10 (paper) -> 5 (now)
decoder_f2_weight = 10 (paper) -> 5 (now)
decoder_f4_weight = 10 (paper) -> 5 (now)
```

# Acknowledgement
This project is built upon numerous previous projects. We'd like to thank the contributors of [STCN](https://github.com/hkchengrex/STCN) and [MiVOS](https://github.com/hkchengrex/MiVOS).

# To do
- [x] quick start and quick evaluation.
- [x] inference codes.
- [x] training detials.
- [x] pre-trained models.







