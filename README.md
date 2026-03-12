# RFLSeg
## Residual Feature Learning for Breast and Thyroid Ultrasound Image segmentation

* **SAIL (Statistic Analysis And Intelligent Learning) Lab of NWU**

* We provide related codes and configuration files to reproduce the "Residual Feature Learning for Breast and Thyroid Ultrasound Image segmentation"


Example images

<p align="center">
  <img src="./img/Fig 2.drawio.png" alt="Image">
</p>


## Introduction
We propose **RFLSeg**, a framework that utilizes a generative network to create residual maps that highlight lesion areas and suppress noise in ultrasound images. By integrating a custom CNN with the SAM encoder, the model leverages these residual features to improve prompt quality and achieve more precise segmentation across multiple ultrasound datasets.

<p align="center">
  <img src="./img/Fig 1.drawio.png" alt="Image">
</p>

## Data Preparation: 
You need to organize the dataset into the following format:

```markdown
dataset_name/
├── img/                
├── label/              
├── splits/             
│   ├── train-{dataset_name}.txt
│   ├── val-{dataset_name}.txt
│   └── test-{dataset_name}.txt      # 1/Breast-BUSI/benign0001 ...
└── ReadMe.txt          

```



## Train the model

prepare the SAM ViT-B weights to `checkpoints/sam_vit_b_01ec64.pth`

**stage 1**: coarse seg
```bash
# self.task = BUSI/UDIAT/TN3K/DDTI
python seg_net.py --mode train
```

```bash
# self.load_model_dict_name = ***.pth
python seg_net.py --mode test --isInfer True
```

**stage 2**: residual map generation
```bash
# self.img_root 
# self.mask_root (coarse_mask)
# self.output_dir 
python sd_infer.py
```

**stage 3**: residual feature learning
```bash
# --task BUSI/UDIAT/TN3K/DDTI
# --preseg_path 
# --gen_name coarse_gen
python train.py
```




## Inference Dataset
```bash
# --task BUSI/UDIAT/TN3K/DDTI
# --preseg_path 
# --gen_name coarse_gen
# --load_cp ***.pth
python test.py
```

## Requirements
  + python3.10
  + pytorch==2.2
  + torchvision==0.15.1
  + numpy
  + Pillow
  + tensorboard
  + pyyaml
  ……