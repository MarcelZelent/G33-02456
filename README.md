# G33-02456
Repository containing our code for the Trackman project. Group 33, for 02456 Deep Learning course in Fall 2024

All of the files can be called from the START_HERE.ipynb file. However, since both our data and computation happens on the HPC, this will not work "straight-out-of-the-box". The results of training are also being sent to Wandb.ai. Please contact the authors for further help.

However, all of the code itself is inside this repository. The general structure is that all models are inside the models.py file with different class names. Then the training for the baseline and parameter tuning can be done though the training class inside modular_test_train.py. The Distilled Knowledge part is implemented with the teacher and student models being inside the models.py file, while training is through distiled_knowledge_training.py. Then the quantization is split into three files: static_quant.py, validation_score.py, validation_score_gpu.py. Feel free to contact the authors for any further questions.

Emails:
Marcel Zelent - s233422@dtu.dk

Filip Roszkowski - s233421@dtu.dk

Karl Mihkel Seenmaa - s232512@dtu.dk
