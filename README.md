# Segment Anything Model (SAM) for Medical Image Segmentation.


*  With the recent introduction of the Segment Anything Model (SAM), this prompt-driven paradigm has entered and revolutionized image segmentation. However, it remains unclear whether it can be applicable to medical image segmentation due to the significant differences between natural images and medical images.
*  In this work, we summarize recent efforts to extend the success of SAM to medical image segmentation tasks and discuss potential future directions for SAM in medical image segmentation, which we hope this work can provide the community with some insights into the future development of foundation models for medical image segmentation.
*  This repo will continue to track and summarize the research progress of SAM in medical image segmentation to boost the research on this topic. If you find this project helpful, please consider stars or citing.

```
@article{SAM4MIS,
  title={How Segment Anything Model (SAM) Boost Medical Image Segmentation?},
  author={Zhang, Yichi and Jiao, Rushi},
  journal={arXiv preprint arXiv:2305.03678},
  year={2023}
}
```

## About Segment Anything Model (SAM)

![image](https://github.com/YichiZhang98/SAM4MIS/blob/main/SAM.jpg)

Segment Anything Model (SAM) uses vision transformer-based image encoder to extract image features and compute an image embedding, and prompt encoder to embed prompts and incorporate user interactions. Then extranted information from two encoders are combined to alightweight mask decoder to generate segmentation results based on the image embedding, prompt embedding, and output token. For more details, please refer to the [original paper](https://arxiv.org/pdf/2304.02643.pdf).


## Literature Reviews of Applying SAM for Medical Image Segmentation.

|Date|Authors|Title|Code|
|---|---|---|---|
|202307|X. Shi et al.|Cross-modality Attention Adapter: A Glioma Segmentation Fine-tuning Method for SAM Using Multimodal Brain MR Images [(paper)](https://arxiv.org/pdf/2307.01124.pdf)|None|
|202307|C. Cui et al.|All-in-SAM: from Weak Annotation to Pixel-wise Nuclei Segmentation with Prompt-based Finetuning [(paper)](https://arxiv.org/pdf/2307.00290.pdf)|None|
|202306|W. Lei et al.|MedLSAM: Localize and Segment Anything Model for 3D Medical Images [(paper)](https://arxiv.org/pdf/2306.14752.pdf)|[Code](https://github.com/openmedlab/MedLSAM)|
|202306|X. Hu et al.|How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images [(paper)](https://arxiv.org/pdf/2306.13731.pdf)|[Code](https://github.com/xhu248/AutoSAM)|
|202306|S. Gong et al.|3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2306.13465.pdf)|[Code](https://github.com/med-air/3DSAM-adapter)|
|202306|S. Chai et al.|Ladder Fine-tuning approach for SAM integrating complementary network [(paper)](https://arxiv.org/pdf/2306.12737.pdf)|[Code](https://github.com/11yxk/SAM-LST)|
|202306|L. Zhang et al.|Segment Anything Model (SAM) for Radiation Oncology [(paper)](https://arxiv.org/pdf/2306.11730.pdf)|None|
|202306|N. Li et al.|Segment Anything Model for Semi-Supervised Medical Image Segmentation via Selecting Reliable Pseudo-Labels [(paper)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4477443)|None|
|202306|G. Ning et al.|The potential of 'Segment Anything' (SAM) for universal intelligent ultrasound image guidance [(paper)](https://www.jstage.jst.go.jp/article/bst/advpub/0/advpub_2023.01119/_pdf)|None|
|202306|C. Shen et al.|Temporally-Extended Prompts Optimization for SAM in Interactive Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2306.08958.pdf)|None|
|202306|T. Shaharabany et al.|AutoSAM: Adapting SAM to Medical Images by Overloading the Prompt Encoder [(paper)](https://arxiv.org/pdf/2306.06370.pdf)|None|
|202306|Y. Gao et al.|DeSAM: Decoupling Segment Anything Model for Generalizable Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2306.00499.pdf)|[Code](https://github.com/yifangao112/DeSAM)|
|202305|D. Lee et al.|IAMSAM : Image-based Analysis of Molecular signatures using the Segment-Anything Model [(paper)](https://www.biorxiv.org/content/biorxiv/early/2023/05/25/2023.05.25.542052.full.pdf)|[Code](https://github.com/portrai-io/IAMSAM)|
|202305|M. Hu et al.|BreastSAM: A Study of Segment Anything Model for Breast Tumor Detection in Ultrasound Images [(paper)](https://arxiv.org/pdf/2305.12447.pdf)|None|
|202305|J. Wu|PromptUNet: Toward Interactive Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2305.10300.pdf)|[Code](https://github.com/WuJunde/PromptUNet)|
|202305|Y. Li et al.|Polyp-SAM: Transfer SAM for Polyp Segmentation [(paper)](https://arxiv.org/pdf/2305.00293.pdf)|[Code](https://github.com/ricklisz/Polyp-SAM)|
|202305|C. Mattjie et al.|Exploring the Zero-Shot Capabilities of the Segment Anything Model (SAM) in 2D Medical Imaging: A Comprehensive Evaluation and Practical Guideline [(paper)](https://arxiv.org/pdf/2305.00109.pdf)|None|
|202305|D. Cheng et al.|SAM on Medical Images: A Comprehensive Study on Three Prompt Modes [(paper)](https://arxiv.org/pdf/2305.00035.pdf)|None|
|202304|A. Wang et al.|SAM Meets Robotic Surgery: An Empirical Study in Robustness Perspective [(paper)](https://arxiv.org/pdf/2304.14674.pdf)|None|
|202304|Y. Huang et al.|Segment Anything Model for Medical Images? [(paper)](https://arxiv.org/pdf/2304.14660.pdf)|None|
|202304|M. Hu et al.|SkinSAM: Empowering Skin Cancer Segmentation with Segment Anything Model [(paper)](https://arxiv.org/pdf/2304.13973.pdf)|None|
|202304|B. Wang et al.|GazeSAM: What You See is What You Segment [(paper)](https://arxiv.org/pdf/2304.13844.pdf)|[Code](https://github.com/ukaukaaaa/GazeSAM)|
|202304|K. Zhang and D. Liu|Customized Segment Anything Model for Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2304.13785.pdf)|[Code](https://github.com/hitachinsk/SAMed)|
|202304|Z. Qiu et al.|Learnable Ophthalmology SAM [(paper)](https://arxiv.org/pdf/2304.13425.pdf)|[Code](https://github.com/Qsingle/LearnablePromptSAM)|
|202304|P. Shi et al.|Generalist Vision Foundation Models for Medical Imaging: A Case Study of Segment Anything Model on Zero-Shot Medical Segmentation [(paper)](https://arxiv.org/pdf/2304.12637.pdf)|None|
|202304|J. Wu et al.|Medical SAM Adapter: Adapting Segment Anything Model for Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2304.12620.pdf)|[Code](https://github.com/WuJunde/Medical-SAM-Adapter)|
|202304|J. Ma and B. Wang|Segment Anything in Medical Images [(paper)](https://arxiv.org/pdf/2304.12306.pdf)|[Code](https://github.com/bowang-lab/MedSAM)|
|202304|Y. Zhang et al.|Input Augmentation with SAM: Boosting Medical Image Segmentation with Segmentation Foundation Model [(paper)](https://arxiv.org/pdf/2304.11332.pdf)|None|
|202304|MA. Mazurowski et al.|Segment Anything Model for Medical Image Analysis: an Experimental Study [(paper)](https://arxiv.org/pdf/2304.10517.pdf)|[Code](https://github.com/mazurowski-lab/segment-anything-medical)|
|202304|S. He et al.|Accuracy of Segment-Anything Model (SAM) in medical image segmentation tasks [(paper)](https://arxiv.org/pdf/2304.09324.pdf)|None|
|202304|T. Chen et al.|SAM Fails to Segment Anything? – SAM-Adapter: Adapting SAM in Underperformed Scenes: Camouflage, Shadow, Medical Image Segmentation, and More [(paper)](https://arxiv.org/pdf/2304.09148.pdf)|[Code](http://tianrun-chen.github.io/SAM-Adaptor/)| 
|202304|C. Hu and X. Li|When SAM Meets Medical Images: An Investigation of Segment Anything Model (SAM) on Multi-phase Liver Tumor Segmentation [(paper)](https://arxiv.org/pdf/2304.08506.pdf)|None|
|202304|F. Putz et al.|The “Segment Anything” foundation model achieves favorable brain tumor autosegmentation accuracy on MRI to support radiotherapy treatment planning [(paper)](https://arxiv.org/pdf/2304.07875.pdf)|None|
|202304|T. Zhou et al.|Can SAM Segment Polyps? [(paper)](https://arxiv.org/pdf/2304.07583.pdf)|[Code](https://github.com/taozh2017/SAMPolyp)|
|202304|Y. Liu et al.|SAMM (Segment Any Medical Model): A 3D Slicer Integration to SAM [(paper)](https://arxiv.org/pdf/2304.05622.pdf)|[Code](https://github.com/bingogome/samm)|
|202304|S. Roy et al.|SAM.MD: Zero-shot medical image segmentation capabilities of the Segment Anything Model [(paper)](https://arxiv.org/pdf/2304.05396.pdf)|None|
|202304|S. Mohapatra et al.|SAM vs BET: A Comparative Study for Brain Extraction and Segmentation of Magnetic Resonance Images using Deep Learning [(paper)](https://arxiv.org/pdf/2304.04738.pdf)|None|
|202304|R. Deng et al.|Segment Anything Model (SAM) for Digital Pathology: Assess Zero-shot Segmentation on Whole Slide Imaging [(paper)](https://arxiv.org/pdf/2304.04155.pdf)|None|
