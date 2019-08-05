# AffineGAN

This is the official pytorch implementions for the paper "Facial Image-to-Video Translation by a Hidden Aine Transformation" in ACM Multimedia 19.

## Installation

- Python 3.6

- Clone this repo:

```
git clone https://github.com/sunlightsgy/AffineGAN.git
cd AffineGAN
```

- Install PyTorch 0.4+(1.1 has been tested) and torchvision from [http://pytorch.org](http://pytorch.org/) and other dependencies (e.g., [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)). 
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`.

## Datasets

### Datasets preparation

- As not all the volunteers agree to make their expressions public, we cannot release the whole CK-Mixed and Cheeks&Eyes dataset. Instead we provide some examples in` ./datasets/`.
- You can create your own dataset following the examples, with the first frame in your training videos is the initial expression. You can download [CK+ dataset](http://www.consortium.ri.cmu.edu/ckagree/) for Gray-scale expression training.
- Note that for **Cheeks&Eyes**, we ask only less than 100 volunteers collect these data using the camera of their cellphone. Thus, the dataset collection is easy. Details can be seen in the paper.
- Some tips for the dataset collection. If you don't follow these tips, you can still train the model, but may harm the performance to some degree.
  - Proper aspect ratio, as we will resize the image to 256*256 when training.
  - Make the head centered in the image.
  - Keep the head fixed. Only the facial expressions are changing. 
  - The attributes of volunteers like race and age will influence the effects of our model. Model trained on Asian people may not perform satisfactorily for European people. 
- You can create new expression categories, like opening eyes. But note that large motions of head (e.g., nodding or shaking head) cannot be well captured by AffineGAN now.
- Please keep the dataset folder structure as follows:

```
dataset_name/
│
├──train/
│  ├──img/
│  │  ├──video_name
│  │  │  ├──0001.jpg
│  │  │  ├──0002.jpg
│  │  │  ├──0003.jpg
│  │  │  └──...
│  └──patch/(if necessary)
│     └──video_name
│        ├──0001.jpg
│        ├──0002.jpg
│        ├──0003.jpg
│        └──...
└──test/
   └──img/  
      └──video_name
         ├──0001.jpg
         ├──0002.jpg
         ├──0003.jpg
         └──...
```

## Usage

We provide sample scripts for training and generation in `./scripts/`.

### Training

- To train a model (The detailed options can be seen in `options/*_options`):

```
python train.py --dataroot /path/to/dataset --name your_exp_name --checkpoints_dir /path/to/checkpoints
```

- If you do not use the local patch for mouse, add option `--no_patch`.
- For continue training, add `--continue_train` and the model will be trained from the latest model.
- The current version only supports training on single GPU, and set batch_size to 1.

### Pretrained Models

We provide some [pretrained models](https://drive.google.com/open?id=1zVhM2VQTirvMQyZmYrjfA5qLf8UdkS54) for all the expressions. Note that it is may not be the optimal one we have, and may perform badly in some online images if they are much different from our training samples. Please place the model in `/path/to/checkpoints`. For example, `/path/to/checkpoints/happy/latest_net_G.pth`, where **happy** is the name of experiment specified in `--name` option.


### Generation

- Download our pretrained models or use your models.
- To generate frames for given test images.

```
python generate.py --dataroot /path/to/dataset --name your_exp_name --checkpoints_dir /path/to/checkpoints --results_dir /path/to/result --eval
```

- Make gifs from generated frames.

```
python img2gif.py --exp_names exp_name1,exp_name1,.. --dataroot /path/to/dataset
```

- You will see the results in the specified `results_dir`.


## Citation

If you use this code for your research, please cite our papers.

```
@inproceedings{shen2019facial,
  title={Facial Image-to-Video Translation by a Hidden Affine Transformation},
  author={Shen, Guangyao and Huang, Wenbing and Gan, Chuang and Tan, Mingkui and Huang, Junzhou and Zhu, Wenwu and Gong, Boqing},
  booktitle={Proceedings of the 27th ACM international conference on Multimedia},
  year={2019},
  organization={ACM}
}
```

## Acknowledgments

This project is heavily borrowed from the project [Image-to-Image Translation in PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Special thanks for [Lijie Fan](http://lijiefan.me/) and the volunteers who provide their expressions.