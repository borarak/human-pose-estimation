## Human-pose-estimation

This repository contains my own experiments, research, reproductions of state-of-the-art human pose estimation. 


## TODO list

- [x] Vanilla hour-glass networks with stacking, trained from scratch (no transfer learning)
- [x] Step/cyclic learning rates (experimental)
- [ ] Gaussian smoothed labels
- [x] More data augumentation
- [x] [Gated skip connections](https://arxiv.org/abs/2002.11098)
- [x] Add data preparation steps
- [ ] Optimise for real-time inference speeds


## Dataset

Currently, i have used the COCO person keypoints dataset to train. Since this is a multi-person dataset, there is a preparation step to extract single persons.

To prepare the data, follow the [prepare_verify_dataset.ipynb](notebooks/prepare_verify_dataset.ipynb) notebook. This notebook crops out every person instance from the COCO dataset using the person bounding boxes and creates new annotations from them. Since our goal is to do single person pose estimation, it's important that each trainng image has one centered person. In some cases, it's not possible to have only person, in that case the algorithm is nudged to concentrate on the centered and focussed person. During inference times, a person object detector can be used first detect each person instance in the image and subsequently feed that detection to the pose model. There are ways to do dense human pose estimation ofcourse but that's not the focus of this repository.


## Train

An example experiment exists in [experiments/hourglass_stack1.sh](experiments/hourglass_stack1.sh)	

1. Replace the directory configurations inside the script. 

2. To train call one of the experiments as follows or create your own

	`./experiments/hourglass_stack1.sh`


## Results


![sample1.png](data/samples/sample1.jpg) 

![sample2.png](data/samples/sample2.jpg)

![sample3.png](data/samples/sample3.jpg)



## Support

In case you are interested in support or collaboration, send me a message on [LinkedIn](https://www.linkedin.com/in/raktim-bora-66832b17/)