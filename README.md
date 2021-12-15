# SetFit

A scikit-learn API version of a SetFit classifier. Model originally developed by [Moshe Wasserblat](https://twitter.com/MosheWasserblat).

## Use

```python
from setfit import SetFitClassifier
docs = ["yay", "boo", "yes", "no", "yeah"]
labels = [1, 0, 1, 0, 1]

# takes a sentence-transformers model
clf = SetFitClassifier("paraphrase-MiniLM-L3-v2")
# fine-tunes embeddings + trains logistic regression head
clf.fit(docs, labels) 

clf.predict(["affirmitive", "negative"])
array([1, 0])
```

## Installation
```pip install git+https://github.com/pmbaumgartner/setfit```

## References
[Original Blog Post](https://moshewasserblat.medium.com/sentence-transformer-fine-tuning-setfit-outperforms-gpt-3-on-few-shot-text-classification-while-d9a3788f0b4e) ([Archived](http://archive.today/Kelkb))

Reference Notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MosheWasserb/SetFit/blob/main/SetFit_SST_2.ipynb)