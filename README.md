# Segment Anything Model (SAM) for Medical Image Segmentation.

*  **[New] We update a new version of annual review of Segment Anything Model for Medical Image Segmentation in 2023. Please refer to [the paper](https://arxiv.org/pdf/2401.03495.pdf) for more details.**

*  Due to the inherent flexibility of prompting, foundation models have emerged as the predominant force in the fields of natural language processing and computer vision. The recent introduction of the Segment Anything Model (SAM) signifies a noteworthy expansion of the prompt-driven paradigm into the domain of image segmentation, thereby introducing a plethora of previously unexplored capabilities. However, the viability of its application to medical image segmentation remains uncertain, given the substantial distinctions between natural and medical images.

*  In this work, we provide a comprehensive overview of recent endeavors aimed at extending the efficacy of SAM to medical image segmentation tasks, encompassing both empirical benchmarking and methodological adaptations. Additionally, we explore potential avenues for future research directions in SAM's role within medical image segmentation. 

*  This repo will continue to track and summarize the latest research progress of SAM in medical image segmentation to support ongoing research endeavors. If you find this project helpful, please consider stars or citing. Feel free to contact for any suggestions.


```
@article{SAM4MIS-2024,
  title={Segment Anything Model for Medical Image Segmentation: Current Applications and Future Directions},
  author={Zhang, Yichi and Shen, Zhenrong and Jiao, Rushi},
  journal={arXiv preprint arXiv:2401.03495},
  year={2024}
}

@article{SAM4MIS-2023,
  title={How Segment Anything Model (SAM) Boost Medical Image Segmentation?},
  author={Zhang, Yichi and Jiao, Rushi},
  journal={arXiv preprint arXiv:2305.03678},
  year={2023}
}
```

## A brief chronology of Segment Anything Model (SAM) and its variants for medical image segmentation in 2023.

![image](https://github.com/YichiZhang98/SAM4MIS/blob/main/timeline.png)



## About Segment Anything Model (SAM)

![image](https://github.com/YichiZhang98/SAM4MIS/blob/main/SAM_v2.jpg)

Segment Anything Model (SAM) uses vision transformer-based image encoder to extract image features and compute an image embedding, and prompt encoder to embed prompts and incorporate user interactions. Then extranted information from two encoders are combined to alightweight mask decoder to generate segmentation results based on the image embedding, prompt embedding, and output token. For more details, please refer to the [original paper](https://arxiv.org/pdf/2304.02643.pdf).



## Large-scale Datasets for Foundation Models for Medical Imaging.

|Date|Authors|Title|Dataset|
|---|---|---|---|
|202311|J. Ye et al.|SA-Med2D-20M Dataset: Segment Anything in 2D Medical Imaging with 20 Million masks [(paper)](https://arxiv.org/pdf/2311.11969.pdf)|[Link](https://openxlab.org.cn/datasets/GMAI/SA-Med2D-20M)|


## Literature Reviews of Applying SAM for Medical Image Segmentation.

|Date|Authors|Title|Code|
|---|---|---|---|
|202401|H. Gu et al.|SegmentAnyBone: A Universal Model that Segments Any Bone at Any Location on MRI [(paper)](https://arxiv.org/pdf/2401.12974.pdf)|[Code](https://github.com/mazurowski-lab/SegmentAnyBone)|
|202401|S. Li et al.|ClipSAM: CLIP and SAM Collaboration for Zero-Shot Anomaly Segmentation [(paper)](https://arxiv.org/pdf/2401.12665.pdf)|[Code](https://github.com/Lszcoding/ClipSAM)|
|202401|JD. Gutiérrez et al.|No More Training: SAM's Zero-Shot Transfer Capabilities for Cost-Efficient Medical Image Segmentation[(paper)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10388320)|None|
|202401|H. Wang et al.|Leveraging SAM for Single-Source Domain Generalization in Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2401.02076.pdf)|[Code](https://github.com/SARIHUST/SAMMed)|
|202401|Z. Feng et al.|Swinsam: Fine-Grained Polyp Segmentation in Colonoscopy Images Via Segment Anything Model Integrated with a Swin Transformer Decoder [(paper)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4673046)|None|
|202312|Z. Zhao et al.|One Model to Rule them All: Towards Universal Segmentation for Medical Images with Text Prompts [(paper)](https://arxiv.org/pdf/2312.17183.pdf)|[Code](https://zhaoziheng.github.io/MedUniSeg)|
|202312|W. Yue et al.|Part to Whole: Collaborative Prompting for Surgical Instrument Segmentation [(paper)](https://arxiv.org/pdf/2312.14481.pdf)|[Code](https://github.com/wenxi-yue/SurgicalPart-SAM)|
|202312|ZM. Colbert et al.|Repurposing Traditional U-Net Predictions for Sparse SAM Prompting in Medical Image Segmentation [(paper)](https://iopscience.iop.org/article/10.1088/2057-1976/ad17a7/meta)|None|
|202312|W. Xie et al.|SAM Fewshot Finetuning for Anatomical Segmentation in Medical Images [(paper)](https://openaccess.thecvf.com/content/WACV2024/papers/Xie_SAM_Fewshot_Finetuning_for_Anatomical_Segmentation_in_Medical_Images_WACV_2024_paper.pdf)|None|
|202312|JG. Almeida et al.|Testing the Segment Anything Model on radiology data [(paper)](https://arxiv.org/pdf/2312.12880.pdf)|None|
|202312|M. Barakat et al.|Towards SAMBA: Segment Anything Model for Brain Tumor Segmentation in Sub-Sharan African Populations [(paper)](https://arxiv.org/pdf/2312.11775.pdf)|None|
|202312|Y. Zhang et al.|SQA-SAM: Segmentation Quality Assessment for Medical Images Utilizing the Segment Anything Model [(paper)](https://arxiv.org/pdf/2312.09899.pdf)|[Code](https://github.com/yizhezhang2000/SQA-SAM)|
|202312|S. Chen et al.|ASLseg: Adapting SAM in the Loop for Semi-supervised Liver Tumor Segmentation [(paper)](https://arxiv.org/pdf/2312.07969.pdf)|None|
|202312|HE. Wong et al.|ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Medical Image [(paper)](https://arxiv.org/pdf/2312.07381.pdf)|[Code](https://scribbleprompt.csail.mit.edu)|
|202312|Y. Zhang et al.|SemiSAM: Exploring SAM for Enhancing Semi-Supervised Medical Image Segmentation with Extremely Limited Annotations [(paper)](https://arxiv.org/pdf/2312.06316.pdf)|None|
|202312|Y. Zhao et al.|Segment Anything Model-guided Collaborative Learning Network for Scribble-supervised Polyp Segmentation [(paper)](https://arxiv.org/pdf/2312.00312.pdf)|None|
|202311|N. Li et al.|Segment Anything Model for Semi-Supervised Medical Image Segmentation via Selecting Reliable Pseudo-Labels [(paper)](https://link.springer.com/chapter/10.1007/978-981-99-8141-0_11)|None|
|202311|X. Wei et al.|I-MedSAM: Implicit Medical Image Segmentation with Segment Anything [(paper)](https://arxiv.org/pdf/2311.17081.pdf)|None|
|202311|Z. Shui et al.|Unleashing the Power of Prompt-driven Nucleus Instance Segmentation [(paper)](https://arxiv.org/pdf/2311.15939.pdf)|[Code](https://github.com/windygoo/PromptNucSeg)|
|202311|M. Li and G. Yang et al.|Where to Begin? From Random to Foundation Model Instructed Initialization in Federated Learning for Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2311.15463.pdf)|None|
|202311|AK. Tyagi et al.|Guided Prompting in SAM for Weakly Supervised Cell Segmentation in Histopathological Images [(paper)](https://arxiv.org/pdf/2311.17960.pdf)|[Code](https://github.com/dair-iitd/Guided-Prompting-SAM)|
|202311|Y. Du et al.|SegVol: Universal and Interactive Volumetric Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2311.13385.pdf)|[Code](https://github.com/BAAI-DCAI/SegVol)|
|202311|DM. Nguyen et al.|On the Out of Distribution Robustness of Foundation Models in Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2311.11096.pdf)|None|
|202311|U. Israel et al.|A Foundation Model for Cell Segmentation [(paper)](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v2.abstract)|[Code](https://label-dev.deepcell.org)|
|202311|Q. Quan et al.|Slide-SAM: Medical SAM Meets Sliding Window [(paper)](https://arxiv.org/pdf/2311.10121.pdf)|None|
|202311|Y. Zhang et al.|Segment Anything Model with Uncertainty Rectification for Auto-Prompting Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2311.10529.pdf)|[Code](https://github.com/YichiZhang98/UR-SAM)|
|202311|Y. Wang et al.|SAMIHS: Adaptation of Segment Anything Model for Intracranial Hemorrhage Segmentation [(paper)](https://arxiv.org/pdf/2311.08190.pdf)|[Code](https://github.com/mileswyn/SAMIHS)|
|202311|H. Jiang et al.|GlanceSeg: Real-time microangioma lesion segmentation with gaze map-guided foundation model for early detection of diabetic retinopathy [(paper)](https://arxiv.org/pdf/2311.08075.pdf)|None|
|202311|Y. Xu et al.|EviPrompt: A Training-Free Evidential Prompt Generation Method for Segment Anything Model in Medical Images [(paper)](https://arxiv.org/pdf/2311.06400.pdf)|None|
|202311|DL. Ferreira and R. Arnaout| Are foundation models efficient for medical image segmentation? [(paper)](https://arxiv.org/pdf/2311.04847.pdf)|[Code](https://github.com/ArnaoutLabUCSF/CardioML)|
|202310|H. Li et al.|Promise:Prompt-driven 3D Medical Image Segmentation Using Pretrained Image Foundation Models [(paper)](https://arxiv.org/pdf/2310.19721.pdf)|[Code](https://github.com/MedICL-VU/ProMISe)|
|202310|D. Anand et al.|One-shot Localization and Segmentation of Medical Images with Foundation Models [(paper)](https://arxiv.org/pdf/2310.18642.pdf)|None|
|202310|H. Wang et al.|SAM-Med3D [(paper)](https://arxiv.org/pdf/2310.15161.pdf)|[Code](https://github.com/uni-medical/SAM-Med3D)|
|202310|SK. Kim et al.|Evaluation and improvement of Segment Anything Model for interactive histopathology image segmentation [(paper)](https://arxiv.org/pdf/2310.10493.pdf)|[Code](https://github.com/hvcl/SAM_Interactive_Histopathology)|
|202310|X. Chen et al.|SAM-OCTA: Prompting Segment-Anything for OCTA Image Segmentation [(paper)](https://arxiv.org/pdf/2310.07183.pdf)|[Code](https://github.com/ShellRedia/SAM-OCTA)|
|202310|M. Peivandi et al.|Empirical Evaluation of the Segment Anything Model (SAM) for Brain Tumor Segmentation [(paper)](https://arxiv.org/pdf/2310.06162.pdf)|None|
|202310|H. Ravishankar et al.|SonoSAM - Segment Anything on Ultrasound Images [(paper)](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_3)|None|
|202310|A. Ranem et al.|Exploring SAM Ablations for Enhancing Medical Segmentation in Radiology and Pathology [(paper)](https://arxiv.org/pdf/2310.00504.pdf)|None|
|202310|S. Pandey et al.|Comprehensive Multimodal Segmentation in Medical Imaging: Combining YOLOv8 with SAM and HQ-SAM Models [(paper)](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Pandey_Comprehensive_Multimodal_Segmentation_in_Medical_Imaging_Combining_YOLOv8_with_SAM_ICCVW_2023_paper.pdf)|None|
|202309|Y. Li et al.|nnSAM: Plug-and-play Segment Anything Model Improves nnUNet Performance [(paper)](https://arxiv.org/pdf/2309.16967.pdf)|[Code](https://github.com/Kent0n-Li/Medical-Image-Segmentation)|
|202309|Y. Zhao et al.|MFS Enhanced SAM: Achieving Superior Performance in Bimodal Few-shot Segmentation [(paper)](https://www.sciencedirect.com/science/article/abs/pii/S1047320323001967)|[Code](https://github.com/VDT-2048/Bi-SAM)|
|202309|C. Wang et al.|SAM-OCTA: A Fine-Tuning Strategy for Applying Foundation Model to OCTA Image Segmentation Tasks [(paper)](https://arxiv.org/pdf/2309.11758.pdf)|[Code](https://github.com/ShellRedia/SAM-OCTA)|
|202309|Y. Zhang et al.|3D-U-SAM Network For Few-shot Tooth Segmentation in CBCT Images [(paper)](https://arxiv.org/pdf/2309.11015.pdf)|None|
|202309|CJ. Chao et al.|Comparative Eminence: Foundation versus Domain-Specific Model for Cardiac Ultrasound Segmentation [(paper)](https://www.medrxiv.org/content/medrxiv/early/2023/09/19/2023.09.19.23295772.full.pdf)|None|
|202309|H. Ning et al.|An Accurate and Efficient Neural Network for OCTA Vessel Segmentation and a New Dataset [(paper)](https://arxiv.org/pdf/2309.09483.pdf)|[Code](https://github.com/nhjydywd/OCTA-FRNet)|
|202309|C. Chen et al.|MA-SAM: Modality-agnostic SAM Adaptation for 3D Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2309.08842.pdf)|[Code](https://github.com/cchen-cc/MA-SAM)|
|202309|P. Zhang and Y. Wang|Segment Anything Model for Brain Tumor Segmentation [(paper)](https://arxiv.org/pdf/2309.08434.pdf)|None|
|202309|B. Fazekas et al.|Adapting Segment Anything Model (SAM) for Retinal OCT [(paper)](https://link.springer.com/chapter/10.1007/978-3-031-44013-7_10)|None|
|202309|X. Lin et al.|SAMUS: Adapting Segment Anything Model for Clinically-Friendly and Generalizable Ultrasound Image Segmentation [(paper)](https://arxiv.org/pdf/2309.06824.pdf)|[Code](https://github.com/xianlin7/SAMUS)|
|202309|X. Xing et al.|SegmentAnything helps microscopy images based automatic and quantitative organoid detection and analysis [(paper)](https://arxiv.org/pdf/2309.04190.pdf)|[Code](https://github.com/XiaodanXing/SAM4organoid)|
|202309|NT. Bui et al.|SAM3D: Segment Anything Model in Volumetric Medical Images [(paper)](https://arxiv.org/pdf/2309.03493.pdf)|[Code](https://github.com/DinhHieuHoang/SAM3D)|
|202308|Y. Zhang et al.|Self-Sampling Meta SAM: Enhancing Few-shot Medical Image Segmentation with Meta-Learning [(paper)](https://arxiv.org/pdf/2308.16466.pdf)|None|
|202308|J. Cheng et al.|SAM-Med2D [(paper)](https://arxiv.org/pdf/2308.16184.pdf)|[Code](https://github.com/uni-medical/SAM-Med2D)|
|202308|C. Li et al.|Auto-Prompting SAM for Mobile Friendly 3D Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2308.14936.pdf)|None|
|202308|W. Feng et al.|Cheap Lunch for Medical Image Segmentation by Fine-tuning SAM on Few Exemplars [(paper)](https://arxiv.org/pdf/2308.14133.pdf)|None|
|202308|Y. Zhang et al.|SamDSK: Combining Segment Anything Model with Domain-Specific Knowledge for Semi-Supervised Learning in Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2308.13759.pdf)|None|
|202308|A. Lou et al.|SAMSNeRF: Segment Anything Model (SAM) Guides Dynamic Surgical Scene Reconstruction by Neural Radiance Field (NeRF) [(paper)](https://arxiv.org/pdf/2308.11774.pdf)|[Code](https://github.com/AngeLouCN/SAMSNeRF)|
|202308|A. Archit et al.|Segment Anything for Microscopy [(paper)](https://www.biorxiv.org/content/10.1101/2023.08.21.554208v1.full.pdf)|[Code](https://github.com/computational-cell-analytics/micro-sam)|
|202308|X. Yao et al.|False Negative/Positive Control for SAM on Noisy Medical Images [(paper)](https://arxiv.org/pdf/2308.10382.pdf)|[Code](https://github.com/xyimaging/FNPC)|
|202308|B. Fazekas et al.|SAMedOCT: Adapting Segment Anything Model (SAM) for Retinal OCT [(paper)](https://arxiv.org/pdf/2308.09331.pdf)|None|
|202308|W. Yue et al.|SurgicalSAM: Efficient Class Promptable Surgical Instrument Segmentation [(paper)](https://arxiv.org/pdf/2308.08746.pdf)|[Code](https://github.com/wenxi-yue/SurgicalSAM)|
|202308|H. Zhang et al.|CARE: A Large Scale CT Image Dataset and Clinical Applicable Benchmark Model for Rectal Cancer Segmentation [(paper)](https://arxiv.org/pdf/2308.08283.pdf)|[Code](https://github.com/kanydao/U-SAM)|
|202308|Q. Wu et al.|Self-Prompting Large Vision Models for Few-Shot Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2308.07624.pdf)|[Code](https://github.com/PeterYYZhang/few-shot-self-prompt-SAM)|
|202308|A. Wang et al.|SAM Meets Robotic Surgery: An Empirical Study on Generalization, Robustness and Adaptation [(paper)](https://arxiv.org/pdf/2308.07156.pdf)|None|
|202308|D. Shin et al.|CEmb-SAM: Segment Anything Model with Condition Embedding for Joint Learning from Heterogeneous Datasets [(paper)](https://arxiv.org/pdf/2308.06957.pdf)|None|
|202308|R. Biswas|Polyp-SAM++: Can A Text Guided SAM Perform Better for Polyp Segmentation? [(paper)](https://arxiv.org/pdf/2308.06623.pdf)|[Code](https://github.com/RisabBiswas/Polyp-SAM++)|
|202308|S. Cao et al.|TongueSAM: An Universal Tongue Segmentation Model Based on SAM with Zero-Shot [(paper)](https://arxiv.org/pdf/2308.06444.pdf)|[Code](https://github.com/cshan-github/TongueSAM)|
|202308|X. Li et al.|Leverage Weakly Annotation to Pixel-wise Annotation via Zero-shot Segment Anything Model for Molecular-empowered Learning [(paper)](https://arxiv.org/pdf/2308.05785.pdf)|None|
|202308|JN. Paranjape et al.|AdaptiveSAM: Towards Efficient Tuning of SAM for Surgical Scene Segmentation [(paper)](https://arxiv.org/pdf/2308.03726.pdf)|[Code](https://github.com/JayParanjape/biastuning)|
|202308|Z. Huang et al.|Push the Boundary of SAM: A Pseudo-label Correction Framework for Medical Segmentation [(paper)](https://arxiv.org/pdf/2308.00883.pdf)|None|
|202307|J. Zhang et al.|SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology [(paper)](https://arxiv.org/pdf/2307.09570.pdf)|None|
|202307|MS. Hossain et al.|Robust HER2 Grading of Breast Cancer Patients using Zero-shot Segment Anything Model (SAM) [(paper)](https://www.preprints.org/manuscript/202307.1213/v1)|None|
|202307|C. Wang et al.|SAM^Med^ : A medical image annotation framework based on large vision model [(paper)](https://arxiv.org/pdf/2307.05617.pdf)|None|
|202307|G. Deng et al.|SAM-U: Multi-box prompts triggered uncertainty estimation for reliable SAM in medical image [(paper)](https://arxiv.org/pdf/2307.04973.pdf)|None|
|202307|H. Kim et al.|Empirical Analysis of a Segmentation Foundation Model in Prostate Imaging [(paper)](https://arxiv.org/pdf/2307.03266.pdf)|None|
|202307|X. Shi et al.|Cross-modality Attention Adapter: A Glioma Segmentation Fine-tuning Method for SAM Using Multimodal Brain MR Images [(paper)](https://arxiv.org/pdf/2307.01124.pdf)|None|
|202307|C. Cui et al.|All-in-SAM: from Weak Annotation to Pixel-wise Nuclei Segmentation with Prompt-based Finetuning [(paper)](https://arxiv.org/pdf/2307.00290.pdf)|None|
|202306|E. Kellener et al.|Utilizing Segment Anything Model for Assessing Localization of Grad-CAM in Medical Imaging [(paper)](https://arxiv.org/pdf/2306.15692.pdf)|None|
|202306|F. Hörst et al.|CellViT: Vision Transformers for Precise Cell Segmentation and Classification [(paper)](https://arxiv.org/pdf/2306.15350.pdf)|[Code](https://github.com/TIO-IKIM/CellViT)|
|202306|W. Lei et al.|MedLSAM: Localize and Segment Anything Model for 3D Medical Images [(paper)](https://arxiv.org/pdf/2306.14752.pdf)|[Code](https://github.com/openmedlab/MedLSAM)|
|202306|X. Hu et al.|How to Efficiently Adapt Large Segmentation Model (SAM) to Medical Images [(paper)](https://arxiv.org/pdf/2306.13731.pdf)|[Code](https://github.com/xhu248/AutoSAM)|
|202306|S. Gong et al.|3DSAM-adapter: Holistic Adaptation of SAM from 2D to 3D for Promptable Medical Image Segmentation [(paper)](https://arxiv.org/pdf/2306.13465.pdf)|[Code](https://github.com/med-air/3DSAM-adapter)|
|202306|DMH. Nguyen et al.|LVM-Med: Learning Large-Scale Self-Supervised Vision Models for Medical Imaging via Second-order Graph Matching [(paper)](https://arxiv.org/pdf/2306.11925.pdf)|[Code](https://github.com/duyhominhnguyen/LVM-Med)|
|202306|S. Chai et al.|Ladder Fine-tuning approach for SAM integrating complementary network [(paper)](https://arxiv.org/pdf/2306.12737.pdf)|[Code](https://github.com/11yxk/SAM-LST)|
|202306|L. Zhang et al.|Segment Anything Model (SAM) for Radiation Oncology [(paper)](https://arxiv.org/pdf/2306.11730.pdf)|None|
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
