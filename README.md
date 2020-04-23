## Human-pose-estimation

This repository contains my own experiments, research, reproductions of state-of-the-art human pose estimation. Please note this is a **WIP** and this repository will be continously updated.


## TODO list

- [x] Vanilla hour-glass networks with stacking, trained from scratch (no transfer learning)
- [x] Step/cyclic learning rates (experimental)
- [ ] Gaussian smoothed labels
- [x] More data augumentation
- [x] Gated skip connections
- [ ] Add data preparation steps
- [ ] Optimise for real-time inference speeds


## Dataset

Currently, i have used the COCO person keypoints dataset to train. Since this is a multi-person dataset, there is a preparation step to extract single persons.

@TODO: Add data preparation step for COCO


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