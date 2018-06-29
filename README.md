- Code will NOT work on CPU. You need an NVIDIA GPU to run this code.
- package versions:
  - matplotlib==2.0.2
  - numpy==1.13.1
  - opencv==3.4.1
  - tensorflow-gpu==1.3.0
  - python==3.6.2


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