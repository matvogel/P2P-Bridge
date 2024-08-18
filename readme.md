<p align="center">

  <h1 align="center">P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising</h1>
  <p align="center">
    <a href="https://matvogel.github.io">Mathias Vogel</a><sup>1</sup>
    <br>
    <a href="https://scholar.google.com/citations?user=ml3laqEAAAAJ">Keisuke Tateno</a><sup>2</sup>,
    <a href="https://inf.ethz.ch/people/person-detail.pollefeys.html">Marc Pollefeys</a><sup>1,3</sup>,
    <a href="https://federicotombari.github.io/">Federico Tombari</a><sup>2,4</sup>,
    <a href="https://scholar.google.com/citations?user=eQ0om98AAAAJ">Marie-Julie Rakotosaona</a><sup>*2</sup>
    <a href="https://francisengelmann.github.io/">Francis Engelmann</a><sup>*1,2</sup>,
    <br>
    <sup>1</sup>ETH Zurich, 
    <sup>2</sup>Google, 
    <sup>3</sup>Microsoft,
    <sup>4</sup>TUM,
    <br>
    <sup>*</sup>Equal Contribution
  </p>
  <h2 align="center">ECCV 2024</h2>
  <h3 align="center"><a href="assets/P2P-Bridge.pdf">📚Paper</a> | <a href="https://github.com/matvogel/P2P-Bridge">💾Code</a> </h3>
  <div align="center"></div>
</p>
<p align="center">
  <a href="">
    <img src="assets/overview.png " width="100%">
  </a>
</p>

<br>

**P2P-Bridge**  introduces a novel approach for point cloud denoising by adapting Diffusion Schrödinger bridges to learn an optimal transport plan between paired point sets. Further enhancements are possible by incorporating additional features such as RGB data and point-wise DINOV2 features.

## ⚙️ Requirements
The code was tested using Python 3.10 and CUDA 11.8 on Ubuntu 22.04 and WSL2. Due to compatibility with older methods, there are quite a few dependencies, but we tried to make installation easier by providing a script and accumulating CUDA code as much as possible.

First, create a new environment (we use [conda](https://conda.io/projects/conda/en/latest/index.html)) and install the dependencies using the following commands:

```console
conda create -n p2pb python=3.10
conda activate p2pb
```
We recommend to first install ``torch`` and ``torchvision`` using the following command:
```console
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia --yes
```
followed by the installation of [Pytorch3D](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md) and [TorchCluster](https://github.com/rusty1s/pytorch_cluster).

Finally, install all other dependencies and compile the custom CUDA code using the following command:

```console
sh install.sh
```	


## 🗂️ Data Preparation
### ⚙️ Requirements
For data preparation, additional libraries are used. The requirements can be installed using the following command from the `data` directory:

```console
pip install -r requirements_data.txt
```

### 🧸 Object Datasets (PU-Net and PC-Net)
Download both zip files from [ScoreDenoise](https://github.com/luost26/score-denoise).
Extract them into `data/objects` such that the folder structure looks as follows:
```console
data/objects
├── examples
├── PCNet
├── PUNet
```	

### 🏠 Indoor Scene Datasets
To prepare the indoor scene datasets, follow the instructions [here](data/readme.md).

## 🚀 Training
We use [wandb](https://wandb.ai/site) to track the training process. To use wandb run

```console
wandb init
```

in the terminal to log into your account (you will be asked for your API key). If you want to disable it, just run
    
```console
wandb disabled
```
before running the training script.

To train a model, adjust the `config` file in the `configs` directory according to your data directory and run the following command:

```console
python train.py --config <CONFIG FILE> --save_dir <SAVE DIRECTORY> --wandb_project <WANDB PROJECT NAME> --wandb_entity <WANDB ENTITY NAME>
```

For all available arguments, run

```console
python train.py --help
```
which will also show you how to train using multiple GPUs.

## 📊 Evaluation
### 🧸 PU-Net and PC-Net
To run an evaluation on the PU-Net and PC-Net test data, run the following two commands to reproduce our paper results. The commands first run the denoising on the test data, followed by metrics calculation.

```console
python evaluate_objects.py --model_path ./pretrained/PVDS_PUNet/latest.pth --dataset PUNet
python evaluate_objects.py --model_path ./pretrained/PVDS_PUNet/latest.pth --dataset PCNet
```
The outputs are stored in `output_objects/<dataset>` together with the metrics. The output folder can be changed using the `--output_root` argument. For all available arguments, run

```console
python evaluate_objects.py --help
```

### 🏠 Indoor Scenes
To reproduce results on the indoor scenes dataset, we provide the following instructions for ScanNet++. The ARKitScenes dataset can be evaluated in the same way.

#### 1. **Denoising:**
To denoise rooms from our ScanNet++ test set, you need to have the rooms specified in `splits/snpp_test.txt` preprocessed (see [here](data/readme.md)). For automatic evaluation, copy all `snpp_test` scenes into a seperate folder called `snpp_evaluation`. Then you can use our script to denoise all test rooms:

```console
sh scripts/denoise_snpp.sh <PATH TO snpp_evaluation>
```

#### 2. **Evaluation:**
To evaluate the denoised rooms, run the following command:
```console
python evaluate_rooms.py --data_root <PATH TO snpp_evaluation> --dataset snpp
```

This will calculate the metrics for all prediction files and generate a `csv` file in the `predictions` folder inside `snpp_evaluation`. Note that the commands above use our coordinate only model by default. If you want to evaluate the models using RGB and DINOV2 features, use the RGB or RGB_DINO checkpoints by providing the corresponding weigths in the `--model_path` argument.

## 🧩 Denoise Your Own Data
<p align="center">
  <a href="">
    <video src="assets/room-denoise.mp4" width="320" height="240" autoplay loop controls></video>
  </a>
</p>

### Real-World Data
To denoise real-world data such as indoor scenes, you can use the following command:

```console
python denoise_room.py --room_path <ROOM PATH> --model_path <MODEL PATH> --out_path <OUTPUT PATH>
```

If you want to use precalculated features use the `--feature_name` argument. For all available arguments, run

```console
python denoise_room.py --help
```

### Synthetic Data (Objects)

To denoise synthetic data, you can use the following command:

```console
python denoise_object.py --data_path <PATH TO XYZ FILE> --save_path <OUTPUT FILE> --model_path <MODEL PATH>   
```
