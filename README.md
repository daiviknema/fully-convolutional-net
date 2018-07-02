## Fully Convolutional Network (Tensorflow Implementation)

A Tensorflow implementation of the landmark paper by Long, Shelhamer and Darrell; which presents Fully Convolutional Networks for semantic segmentation ([link to paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)). While better methods for semantic segmentation exist today, this was the first paper to successfully use end-to-end neural network training to segment images.

I think that this is probably more for my own benefit than anyone else - but I've still tried to make to code as readable as possible in case someone else finds this and wants to play around with it. Detailed implementation notes can be found in [this blog post](https://plantsandbuildings.github.io/machine-learning/deep-learning/computer-vision/2018/06/29/semantic-segmentation-using-fully-convolutional-networks.html).

### Sample results produced by this code:

| Original Image | Ground Truth Segmentation | Predicted Segmentation |
| :------: | :---------: | :--------: |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/1.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/1gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/1p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/2.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/2gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/2p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/3.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/3gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/3p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/4.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/4gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/4p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/5.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/5gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/5p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/6.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/6gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/6p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/7.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/7gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/7p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/8.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/8gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/8p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/9.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/9gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/9p.png) |
| ![Original Image](https://plantsandbuildings.github.io/static/img/fcn/results/10.png) | ![Ground Truth Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/10gt.png) | ![Predicted Segmentation](https://plantsandbuildings.github.io/static/img/fcn/results/10p.png) |


### Prerequisites for building and running the model

- Code will NOT work on a CPU. You need an NVIDIA GPU to run this code. I used a [Paperspace](https://www.paperspace.com/) VM (32 GB RAM, 8 GB NVIDIA Quadro M4000 GPU).
- Package versions:
  - matplotlib==2.0.2
  - numpy==1.13.1
  - opencv==3.4.1
  - tensorflow-gpu==1.3.0
  - python==3.6.2
- You will also need to download the [PASCAL-VOC Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) and trained [VGG-16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) weights.

### Training the model and performing inference

The training script `train.py` can be executed as:

``` bash
python train.py NUMBER_OF_COARSE_ITERATIONS NUMBER_OF_FINE_ITERATIONS SAVE_PARAMS_AFTER [RESTORE_CHECKPOINT]
```

The parameters to the program are:

- `NUMBER_OF_COARSE_ITERATIONS`: The number of iterations for which coarse training should run. Coarse training refers to updating only the score layers.
- `NUMBER_OF_FINE_ITERATIONS`: The number of iterations for which fine training should run. Fine training refers to updating all the weights of the network.
- `SAVE_PARAMS_AFTER`: This is the number of iterations after which the performance of the network is evaluated, and if it performs better than ever before, the parameters are saved.
- `RESTORE_CHECKPOINT`: If you want the training to resume from a previous training session, provide the path to the TF checkpoint.

#### Example of training

Example execution (100 coarse iterations, 200 fine iterations, save best params after every 10 iterations):
``` bash
python train.py 100 200 10
```

Now suppose we want to use the best fine training parameters from the previous training session and run an additional 100 fine iterations:
``` bash
python train.py 0 100 10 best_params_fine/fcn_<PQR>.ckpt
```

where <PQR> is the best params index from the previous training session.

The inference script can be run as:

``` bash
python infer.py CHECKPOINT NUM_EXAMPLES
```

The parameters are:

- `CHECKPOINT`: These are the parameters of the model that will be used to perform inference.
- `NUM_EXAMPLES`: Number of example images to perform inference on.



