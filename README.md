# Q-VAM: Query-Guided Visual Attention Modulation for Medical Visual Question Answering

## Overview
Visual prompting has been explored to enhance the ability of vision-language models to perceive visual information, proving particularly effective for fine-grained tasks such as those involving medical images. However, adapting general-domain visual prompting methods to the medical field remains challenging due to the substantial gap between general- and medical-domain data. To this end, we propose Query-Guided Visual Attention Modulation (Q-VAM), an end-to-end framework designed to improve performance on medical visual question answering (VQA) by dynamically modulating visual attention in a task-aware and query-driven manner. Specifically, we introduce a cross-context fusion module that utilizes the early layers of a large language model (LLM) to generate vision-aware query embeddings, capturing both the semantic intent of medical questions and relevant visual context. These enriched embeddings serve as queries to guide both spatial- and channel-wise attention. Spatial attention highlights salient image regions via scaled cross-attention followed by average pooling and is injected into the visual features using a residual strategy that preserves the original visual information while emphasizing task-relevant areas. In contrast, channel attention is computed from the aggregated query embedding through a lightweight network and is applied multiplicatively to reweight feature channels directly. The resulting modulated visual tokens, together with the question tokens, are fed into the language model to generate the final answer. Experimental results on public medical VQA datasets demonstrate the effectiveness of Q-VAM in improving both performance and interpretability.

![Image text](https://github.com/DeepModeler/Q-VAM/blob/main/image/Fig1.png)

## Preparation Before Training and Evaluation
# Create directories
Before starting training and evaluating the model, you need to create two folders named `save_model` and `save_result`. These directories will be used to store the trained models and the generated results, respectively.

# Downoad datasets
The dataset can be downloaded from the following links:
* [Med-VQA](https://drive.google.com/file/d/1l9hnxa2Y3D8rhNLldtCQ0vGPhsiWH_Su/view?usp=sharing) 


## Train and Evaluate Q-VAM Model
To train Q-VAM, run:
```
python run_trainloop.py
```
To evaluate Q-VAM, run:
```
python run_generate.py
```

## Acknowlegements
Our project references the codes in the following repos.
* [Med-MoE](https://github.com/jiangsongtao/Med-MoE)
* [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
