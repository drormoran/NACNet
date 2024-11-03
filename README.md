# 
<p align="center">
  <h1 align="center"> Consensus Learning with Deep Sets for <br> 
    Essential Matrix Estimation <br> 
    <sub><sup> NeurIPS 2024 </sup></sub> </h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=kS5jfSoAAAAJ&hl">Dror Moran</a>
    ·
    <a href="https://scholar.google.com/citations?user=x-L1hsAAAAAJ&hl">Yuval Margalit</a>
    ·
    <a href="https://scholar.google.com/citations?user=Jp8GUPYAAAAJ&hl">Guy Trostianetsky</a>
    ·
    <a href="https://scholar.google.com/citations?user=ucCJtZ8AAAAJ&hl">Fadi Khatib</a>
    ·
    <a href="https://www.weizmann.ac.il/math/meirav/home">Meirav Galun</a>
    ·
    <a href="https://www.weizmann.ac.il/math/ronen/home">Ronen Basri</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2406.17414" align="center">Paper</a> 
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://github.com/drormoran/DeepOulierRemoval/blob/public_version/assets/Model_outputs.jpg" alt="example" width=80%>
    <br>
</p>
NACNet is a Noise-Aware Deep Sets framework to estimate relative camera pose, given a set of putative matches extracted from two views of a scene. We demonstrated that a position denoising of inliers and noise-free pretraining enable accurate estimation of the essential matrix. Our experiments indicate that our method can handle large numbers of outliers and achieve accurate pose estimation superior to previous methods.


## Setup
### Conda environment
Create a conda environment using the following command (tested with CUDA 11.8):
```
conda create -n NACNet python=3.9
conda activate NACNet
pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install lightning
pip install opencv-python kornia plotly ConfigArgParse pandas scipy h5py openpyxl joblib pyparsing protobuf certifi urllib3==2.2.3
```

### Data:
We used [OANet](https://github.com/zjhthu/OANet)'s code for dataset preprocessing. \
Please follow their instructions for downloading and preparing the data.


## Evaluate trained models:
Download the trained models from [here](https://www.dropbox.com/scl/fo/ds55kqu7ps3198f3bj8f1/AEAivElNRj4pKjB8JwNl3ko?rlkey=k3nydltkm4i1v7yehgegj3dx1&st=zqfgdzcg&dl=0). Place the Experiments folder in the repository's main folder.

Evaluate our models by running the following Python script:

```
python eval_runs.py --conf [run configuration file] --version [version] --data_path [data path] 
```
Where `conf` is the configuration file associated with the dataset and descriptor type, `version` is the experiment version, and `data_path` is the path to the folder containing the `.hdf5` file created in the [setup](#setup) section.

For example, for evaluating our model on YFCC with SIFT descriptors run:
```
python eval_runs.py --conf conf/train_yfcc_sift.conf --version final --data_path [data path]
```

## Train a model
We train our model by applying a two-stage, noise-aware optimization process. The input to the first stage includes the set of noise-free inlier matches along
with the outlier matches. In the second stage, the input to the network includes the original set of keypoints.

### First stage training (Noise-free):
```
python train.py --conf conf/noise_free_trainnig.conf --run_name [run_name] --data_type [YFCC \ SUN3D] --desc_name [sift-2000 \ super-2000] --data_path [data path]
```
Where `run_name` is the name of the experiment, `data_type` is the dataset (YFCC or SUN3D), `desc_name` is the descriptor type (SIFT or SuperPoint), and `data_path` is the path to the folder containing the `.hdf5` file created in the [setup](#setup) section.

For example, for training a model on YFCC with SIFT descriptors run:
```
python train.py --conf conf/noise_free_trainnig.conf --run_name noise_free_yfcc_sift --data_type YFCC --desc_name sift-2000 --data_path [data path]
```
To fine-tune a trained model for a new dataset, use the `pretrained_model` argument and specify the trained model path.


### Second stage training:
```
python train.py --conf [run configuration file] --data_path [data path] --pretrained_model [path to noise-free model]
```
Where `conf` is the configuration file associated with the dataset and descriptor type, `data_path` is the path to the folder containing the `.hdf5` file created in the [setup](#setup) section, and `pretrained_model` is the path to the saved model we trained on the noise-free dataset at the first training stage.

For example, for training a model on YFCC with SIFT descriptors run:
```
python train.py --conf conf/train_yfcc_sift.conf --data_path [data path] --pretrained_model noise_free_yfcc_sift/[version]/chckpt/[ckpt file]
```

## Citation
If you find this work useful, please cite:
```
@article{moran2024consensus,
  title={Consensus Learning with Deep Sets for Essential Matrix Estimation},
  author={Moran, Dror and Margalit, Yuval and Trostianetsky, Guy and Khatib, Fadi and Galun, Meirav and Basri, Ronen},
  journal={arXiv preprint arXiv:2406.17414},
  year={2024}
}
```


