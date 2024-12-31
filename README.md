## classical_chinese_poetry_sentiment_analysis
This is the source code for paper
## Abstract
Sentiment analysis of classical Chinese poetry is essential for understanding Chinese ancient literature. The majority of previous work focuses on interpreting the textual meanings to analyze the emotional expressions in poetry, often overlooking the unique prosodic and visual features inherent in poetry. In this work, we propose a multimodal framework for Classical Chinese Poetry Sentiment Analysis. Specifically, we extract sentence-level phonetic features from the poetic sentences and further incorporate regional dialect features to enrich the overall audio features. We also generate visual features for the poetry and the multimodal features are fused with the textual modality features enhanced by LLM translation through multimodal contrastive representation learning. Experimental results on two public datasets demonstrate that our method surpasses the state-of-the-art approaches.
## Prerequisites
The code has been successfully tested in the following environment.
 - Python 3.8.2
 - PyTorch 1.7.0+cu110
 - numpy 1.19.3
 - Sklearn 1.0.2
 - Pandas 1.2.2
 - Transformers 4.3.2
 - pypinyin 0.49.0
 - pinyinsplit 0.1.4
## Getting Started
### Prepare your data
Use data_preprocessing.ipynb to prepare data.
### Training Model
Please run following commands for training.
```python
python fspc_model.py
```
## Cite
Please cite our paper if you find this code useful for your research:
