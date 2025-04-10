# Evaluating Subpopulation Data Poisoning Attacks on Machine Learning Models

 Repository for 4th year Dissertation: Evaluating Subpopulation Data Poisoning Attacks on Machine Learning Models

 Felix Hoare
 s2050013

 ## Data

Three datasets are used in experiments. All processing is achieved in their respective files, which are listed below:

CIFAR-10
Source: https://www.cs.toronto.edu/~kriz/cifar.html
Processing: cifar10_eddie.py for full experimentation, cifar10_paper_imp.ipynb for notebook version
Processed through dataset splitting into train, aux, test, then aux is poisoned and added back into train
Tested on ClusterMatch and ColourSegmentation attacks (colour_segmentation.py, kmeans_colour_segmentation.ipynb)

UCI Adult
Source: https://archive.ics.uci.edu/dataset/2/adult
Processing: uci_adult_attack.ipynb for all experiments
Processed through dataset splitting into train, aux, test, then aux is poisoned and added back into train
Tested on FeatureMatch and ClusterMatch attacks

UTKFace
Source: https://susanqq.github.io/UTKFace/
Processing: utkface_attack.py for full experimentation, utk_face_attack.ipynb for notebook version
Processed through dataset splitting into train, aux, test, then aux is poisoned and added back into train
Tested on FeatureMatch and ClusterMatch attacks

Running the files cifar10_eddie.py, colour_segmentation.py, uci_adult_attack.ipynb, utkface_attack.py will produce the outputs files used in the project
Running the output files through result_analysis.ipynb will produce visualisations for the results