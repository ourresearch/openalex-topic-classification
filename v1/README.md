### v1 of the OpenAlex Topic Classification Model

This directory gives all the code used in creating the OpenAlex Topic Classification Model which was based on labeled data from CWTS. To learn more about how the data was labeled, feel free to visit the following link: [An open approach for classifying research publications](https://www.leidenmadtrics.nl/articles/an-open-approach-for-classifying-research-publications)

If you would like to learn more about how the data was prepared, how the model was created, or how the model was deployed, you can explore the notebooks/code in this directory. 

#### 001 Exploration and Data Processing

Use the notebook in this directory to look into how the labeled data was explored and also how the data was transformed into the features used to train the model. These notebooks using both Spark (Databricks) and Jupyter notebooks so make sure you are in the right environment.

#### 002 Modeling and Testing

Use the notebooks in this directory if you would like to train a model from scratch using the same methods as OpenAlex.

#### 003 Deploy

Use the notebooks in this directory if you would like to deploy the model locally or in AWS. The model artifacts will need to be downloaded into the appropriate folder before the model can be deployed, so make sure to follow the instructions in the "NOTES" section below.


### NOTES
#### Model Artifacts
In order to deploy the model/container in AWS or deploy the model/container locally, you will need the model artifacts. These can be downloaded from Zenodo at the following link: [OpenAlex Topic Classification Model Artifacts and Training Data](https://zenodo.org/records/10568402)

These files will contain the all training data (train/val/test in one dataset), the mapping dictionaries for labels, and the model weights for the full model. In order to make use of these, follow along in the notebooks in the 002 Model section or the deployment code in 003 (model_to_api/container/topic_classifier/predictor.py).

#### Full labeled dataset from CWTS
The initial labeled dataset that was used for initial testing and also to train the model can be found at the following link: [Classification of research publications based on data from OpenAlex](https://zenodo.org/records/10560276)
