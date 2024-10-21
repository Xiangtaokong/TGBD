## Toward Generalizing Visual Brain Decoding to Unseen Subjects

<a href='https://arxiv.org/abs/2410.14445'><img src='https://img.shields.io/badge/arXiv-2410.14445-b31b1b.svg'></a> &nbsp;&nbsp;

Authors: Xiangtao Kong, Kexin Huang, [Ping Li](https://scholar.google.com/citations?user=Z0mAYS4AAAAJ&hl=en&oi=ao) and [Lei Zhang](https://scholar.google.com/citations?user=tAK5l1IAAAAJ&hl=en&oi=ao)

## Note: the project has not been updated yet, please wait a few days before pulling.

## Abstract
Visual brain decoding aims to decode visual information from human brain activities. Despite the great progress, one critical limitation of current brain decoding research lies in the lack of generalization capability to unseen subjects. Prior works typically focus on decoding brain activity of individuals based on the observation that different subjects exhibit different brain activities, while it remains unclear whether brain decoding can be generalized to unseen subjects. This study is designed to answer this question. We first consolidate an image-fMRI dataset consisting of stimulus-image and fMRI-response pairs, involving 177 subjects in the movie-viewing task of the Human Connectome Project (HCP). This dataset allows us to investigate the brain decoding performance with the increase of participants. We then present a learning paradigm that applies uniform processing across all subjects, instead of employing different network heads or tokenizers for individuals as in previous methods, which can accommodate a large number of subjects to explore the generalization capability across different subjects. We conduct a series of experiments and find the following: First, the network exhibits clear generalization capabilities with the increase of training subjects. Second, the generalization capability is common to popular network architectures (MLP, CNN and Transformer). Third, the generalization performance is affected by the similarity between subjects. Our findings reveal the inherent similarities in brain activities across individuals. With the emerging of larger and more comprehensive datasets, it is possible to train a brain decoding foundation model in the future.

:star: If TGBD is helpful to your images or projects, please help star this repo. Thanks! :hugs:

## 🔎 Overview

![Demo Image](https://github.com/Xiangtaokong/TGBD/blob/main/demo_img/pipeline.png)

We present a learning paradigm that applies uniform processing across all subjects, instead of employing different network heads or tokenizers for individuals as in previous methods, which can accommodate a large number of subjects to explore the generalization capability across different subjects.

![Demo Image](https://github.com/Xiangtaokong/TGBD/blob/main/demo_img/line.png)

The network exhibits clear generalization capabilities with the increase of training subjects.


## 📷 Visual Results

![Demo Image](https://github.com/Xiangtaokong/TGBD/blob/main/demo_img/retrieval.png)

## ⚙️ Dependencies and Installation
```
## git clone this repository
git clone https://github.com/Xiangtaokong/TGBD.git
cd TGBD

# create an environment
conda env create -f environment.yml
```

## 🚀 Test

#### Setp 1 Download the pre-trained models



#### Setp 2 Download the testsets


#### Setp 3 Edit the test yml file


#### Setp 4 Run the command

## :star: Train 

#### Step1: Download the training data


#### Step2: Data prepare


#### Step3: Edit the train yml file


#### Step4: Run the command


## ❤️ Acknowledgments

## 📧 Contact
If you have any questions, please feel free to contact: `xiangtao.kong@connect.polyu.hk`

## 🎓Citations
If our code helps your research or work, please consider citing our paper.
The following are BibTeX references:

```

```

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).




<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=Xiangtaokong/MiOIR)

</details>


