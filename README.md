# APT: Attention Prompt Tuning

![VideoMAE Framework](figs/apt.png)


> [**APT: Attention Prompt Tuning**](https://arxiv.org/abs/xxxx)<br>
> [Wele Gedara Chaminda Bandara](https://github.com/wgcban), [Vishal M Patel](https://engineering.jhu.edu/vpatel36/team/vishalpatel/)<br>Johns Hopkins University

## Getting Started

### Conda Environment

Setup the virtual conda environment using the `environment.yml`:
```
conda env create -f environment.yml
```

Then activate the conda environment:
```
conda activate apt
```

### Download the VideoMAE Pre-trained Models:
Download the corresponding VideoMAE pre-trained models given in the `MODEL_ZOO.md`. 

We use VideoMAE pretrianed on Kinetics-400 dataset for our experiments.



## ✏️ Citation

If you think this project is helpful, please feel free to leave a star⭐️ and cite our paper:

```
@inproceedings{tong2022videomae,
  title={Video{MAE}: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Zhan Tong and Yibing Song and Jue Wang and Limin Wang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={arXiv preprint arXiv:2203.12602},
  year={2022}
}
```
