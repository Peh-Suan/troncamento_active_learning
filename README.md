# Troncamento active learning
---

An implementation of active learning for identifying troncamento tokens from [the Common Voice dataset](https://commonvoice.mozilla.org/fr) @ [Breiss Lab](https://www.cbreiss.com) at USC

## Model structure and data
The model takes in the target phone's acoustics, as feature embeddings extracted from a Wav2Vec2 [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53), the preceding phone's acoustics, and the target phone and preceding phone identities, vectorized into phone embeddings.

## Training process
1. Pretraining
Silver labels are previded based on phonologically unambiguous sonorants (/n, m, l, r/) and the target vowel /e/ (i.e., no preceding sonorants) in word-final positions.
After pretraining, supervised active learning can be conducted.

2. Active learning
- First extract the tokens to be labeled based on an uncertainty function. The current function is a based on entropy.
- A "gold_label.csv" file will be created. Label the file with 0 (the phone is not the target /e/) or 1 (the phone is /e/).
- Train the model on the labeled data.
- Repeat the previous three steps until the performance reaches the desired level.

## TextGrid labeling
TextGrid labeling is required. A folder `it_vxc_textgrids17_acoustic17` is expected, with the textgrid files named as `common_voice_it_*.TextGrid`