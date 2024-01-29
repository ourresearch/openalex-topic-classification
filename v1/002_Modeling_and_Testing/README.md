### Files for Exploration

* 001 - Fine-tuning a language model based on title and abstract
* 002 - Training a full model that integrates the fine-tuned language model
* 003 - Quick testing of the results
* 004 - Spark notebook that looks at distribution of labeled data vs predicted data
* 005 - Creating the gold citations file that is used in production (will be updated)
* 006 - Some code that was used to test model on CPU to find the best setup


#### Fine-tuned Multilingual BERT Model

Since the language model was fine-tuned using HuggingFace, we uploaded the model to the hub so that anyone can download the model and use it to assign a topic to a title/abstract. This will only be a part of the final model we will be using in OpenAlex but if you have a large number of works with only a title and an abstract and wanted an easy way to deploy the model, going to HuggingFace will be the easiest way to do so:

[OpenAlex Title/Abstract Only Model](https://huggingface.co/OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract)

The model was trained using the following input data format (so it is recommended the data be in this format as well):

* Using both title and abstract: "<TITLE> {insert-processed-title-here}\n<ABSTRACT> {insert-processed-abstract-here}"
* Using only title: "<TITLE> {insert-processed-title-here}"
* Using only abstract: "<TITLE> NONE\n<ABSTRACT> {insert-processed-abstract-here}"

If you use python, the following code is all you need to download the title/abstract only model and assign topics:

```
from transformers import pipeline

title = "{insert-processed-title-here}"
abstract = "{insert-processed-abstract-here}"

classifier = \
    pipeline(model="OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract", top_k=10)

classifier(f"""<TITLE> {title}\n<ABSTRACT> {abstract}""")
```

This will return the top 10 outputs from the model. There will be 2 pieces of information here:

1. Full Topic Label: Made up of both the [OpenAlex](https://openalex.org/) topic ID and the topic label (ex: "1048: Ecology and Evolution of Viruses in Ecosystems")
2. Model Score: Model's confidence in the topic (ex: "0.364")