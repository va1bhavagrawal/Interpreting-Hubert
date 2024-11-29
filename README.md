# Interpreting Hubert 

We follow the following paper: https://www.isca-archive.org/interspeech_2022/wells22_interspeech.pdf


There are two major tenets of our study:
* Analyzing phonetic relationships in pre-trained HuBERT features. 
* Effect of fine-tuning HuBERT on the phonetic understanding in the features. 

To align corpora from the VCTK dataset (must be pre-downloaded and stored), run
``` bash
python3 align_corpus.py --root_wav_dir <ROOT_WAV_DIR> --root_txt_dir <ROOT_TXT_DIR> --root_corpus_dir <ROOT_CORPUS_DIR> --root_textgrid_dir <ROOT_TEXTGRID_DIR> --speakers p376 
```

This script uses `Montreal Forced Aligner` for alignment. 

## Analyzing pre-trained HuBERT:

For generating aligned phoneme ground truths (from the TextGrid files) and HuBERT clusters, run
```bash
python3 get_hubert_units.py
``` 

For only storing the embeddings of the HuBERT (not the clusters), run
``` bash 
python3 get_hubert_features.py 
```

To visualize the TSNE plots, run 
```
python3 vis_tsne.py 
```

To visualize the confusion matrices, run
```
python3 vis_confusion.py
```

To visualize a PCA plot of the features, run
```
python3 vis_pca.py
```

## Effect of fine-tuning HuBERT.

To train HuBERT on some speaker(s)' data, run
```
python3 train.py 
```
This program will also save the checkpoints for the fine-tuned model. 

To learn a K-Means on the new HuBERT features.
```
python3 learn_kmeans.py  
```
To visualize the confusion matrices for the fine-tuned HuBERT, run
```
python3 get_hubert_units_personalized.py 
``` 
