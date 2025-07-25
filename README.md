<div align="center">

<h1>Dual Visual–Textual Graph Construction for Open-Vocabulary Semantic Segmentation</h1>

</div>

## Abstract
> *Open-vocabulary semantic segmentation aims to assign pixel-level labels using arbitrary text queries, but existing CLIP-based methods often produce diffuse similarity maps and struggle with precise boundaries or small objects. Two-stage approaches—first generating class-agnostic mask proposals, then aligning them to text embeddings—improve shape fidelity but suffer from over-segmentation when using dense SAM proposals and from coarse boundaries when using patch-based spectral clustering. To address these limitations, we propose a novel graph-construction pipeline in which nodes correspond to SAM’s class-agnostic masks—providing fine boundary precision—and are augmented with textual nodes representing the target class embeddings. Edge weights capture both inter‐mask visual similarity and visual–textual affinity via cosine similarity in a joint embedding space, ensuring that clusters reflect full-object semantics rather than only discriminative parts. Our dual visual–textual graph promotes geometrically coherent, semantically complete mask proposals without the need for costly post-processing refinements. With the proposed method, state-of-the-art performance is attained over the latest approaches on several benchmarks.*

## Dependencies and Installation


```
# git clone this repository
git clone https://github.com/AnonymousNeurIPS2025/neurips2025-anon.git
cd dvgt

# create new anaconda env
conda create -n dvgt python=3.10
conda activate dvgt

# install torch and dependencies
pip install -r requirements.txt
```

## SAM

Download the SAM model as follows:

```bash
wget -P model https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Datasets
We include the following dataset configurations in this repo: 
1) `With background class`: PASCAL VOC, PASCAL Context, PASCAL Context 459 (PC459), Cityscapes, ADE20k, ADE847, and COCO-Stuff164k, 
2) `Without background class`: VOC20, Context59 (i.e., PASCAL VOC and PASCAL Context without the background category), and COCO-Object.

For PASCAL Context 459, please follow the [CAT-Seg](https://github.com/KU-CVLAB/CAT-Seg/tree/main/datasets) to prepare the datasets.
For the other datasets, please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets. 
The COCO-Object dataset can be converted from COCO-Stuff164k by executing the following command:

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

## Quick Inference
```
python demo.py
```


## Model evaluation
Please modify some settings in `configs/base_config.py` before running the evaluation.

Single-GPU:

```
python eval.py --config ./config/cfg_DATASET.py --workdir YOUR_WORK_DIR
```

Multi-GPU:
```
bash ./dist_test.sh ./config/cfg_DATASET.py
```

Evaluation on all datasets:
```
python eval_all.py
```

## Citation

```

```

## License
This project is licensed under <a rel="license" href="https://github.com/mc-lan/SmooSeg/blob/master/LICENSE">NTU S-Lab License 1.0</a>. Redistribution and use should follow this license.


## Acknowledgement
This study is supported under the RIE2020 Industry Align- ment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s).

This implementation is based on [OpenCLIP](https://github.com/mlfoundations/open_clip), [SCESAME](https://github.com/ymgw55/SCESAME), [PixelCLIP](https://github.com/cvlab-kaist/PixelCLIP) and [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP). Thanks for the awesome work.

