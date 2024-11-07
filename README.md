# Semantic-Guided Multimodal Sentiment Decoding with Adversarial Temporal-Invariant Learning 
-----------------------------------------------------------------------------------------------------------------------------------
Code for Semantic-Guided Multimodal Sentiment Decoding with Adversarial Temporal-Invariant Learning  (SATI).  
Our checkpints can be download from [here](https://drive.google.com/drive/folders/11umrB8wphhYgMyBPAU7q5MXQ1yOepd0s?usp=drive_link).

## **Data Download**
- Install [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK). Ensure, you can perform from mmsdk import mmdatasdk.
- Option 1: Download [pre-computed splits](https://drive.google.com/drive/folders/1IBwWNH0XjPnZWaAlP1U2tIJH6Rb3noMI) and place the contents inside datasets folder.
- Option 2: Re-create splits by downloading data from MMSDK. For this, simply run the code as detailed next.

## **Running the code**
- cd src
- Set word_emb_path in config.py to [glove file](http://nlp.stanford.edu/data/glove.840B.300d.zip) and [roberta](https://drive.google.com/file/d/1KsZGuAP_s68WyU3wOZ2hZ7vcf7HB0zj3/view?usp=drive_link) path.
- Set sdk_dir to the path of CMU-MultimodalSDK.
- python train.py --data mosi. Replace mosi with mosei or ur_funny for other datasets.

