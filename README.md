# SIP
Code for paper: Partial Multi-View Multi-Label Classification via Semantic Invariance Learning and Prototype Modeling


You can run 'python train.py' for a demo!
The original link to the five datasets is not available now, you can download the datesets and pre-processed incomplete datesets from [here](https://drive.google.com/drive/folders/1ey17GpSJEYpYchY6Du_AOj5Yzi2Ml7JU?usp=drive_link). 

# Typo correction:

1. In the denominator of Eq.(1), "u=v" should be "u!=v".

2. It should be {D_{v}: (S^{(v)}+O^{(v)}) \rightarrow \bar{X}^{(v)}}, instead of {{D}_{v}: S^{(v)} \rightarrow \bar{X}^{(v)}}. The input of decoder D_{v} is the sum of shared and private features. 

If this code is helpful to you, please cite the following paper:
```bibtex
@inproceedings{liu2024masked,
  title={Masked Two-channel Decoupling Framework for Incomplete Multi-view Weak Multi-label Learning},
  author={Liu, Chengliang and Wen, Jie and Liu, Yabo and Huang, Chao and Wu, Zhihao and Luo, Xiaoling and Xu, Yong},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```


Please get in touch with me if you have any questions about running this code!
liucl1996@163.com