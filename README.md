
PyTorch implementation for paper [Hierarchical Attention Fusion with Synergistic Adversarial Contrastive Learning for Incomplete Multi-View Clustering]


## Requirements

pytorch>=1.11.0 

numpy>=1.23.4

munkres>=1.1.4

## Datasets

You could find the dataset we used in the paper at './data/...''.


## Training

The hyper-parameters, the training options are defined in the configure file.


~~~bash
run.py --config_file=config/aloideep3v.yaml
~~~

~~~bash
run.py --config_file=config/YouTubeFace50.yaml
~~~

