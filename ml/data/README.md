# Emotion training data

`goemotions_six.csv` is a derived subset of **GoEmotions: A Dataset of Fine-Grained Emotions** by Demszky et al. (ACL 2020). The source dataset contains Reddit comments annotated by human raters and is distributed by Google Research under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Source: https://github.com/google-research/google-research/tree/master/goemotions

Citation:

```bibtex
@inproceedings{demszky2020goemotions,
  author = {Demszky, Dorottya and Movshovitz-Attias, Dana and Ko, Jeongwoo and Cowen, Alan and Nemade, Gaurav and Ravi, Sujith},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  title = {GoEmotions: A Dataset of Fine-Grained Emotions},
  year = {2020}
}
```

Run `python ml/prepare_goemotions.py` to reproduce the balanced derivative. The script excludes unclear or cross-group annotations and maps related fine-grained labels into SpotiSense's six product categories. This transformation changes the original label taxonomy and should be considered when interpreting evaluation results.
