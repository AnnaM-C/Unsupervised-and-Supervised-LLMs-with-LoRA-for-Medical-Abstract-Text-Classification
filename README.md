# Unsupervised-and-Supervised-LLMs-with-LoRA-for-Medical-Abstract-Text-Classification

In the medical field, diagnosing diseases accurately and early is crucial but challenging due to the complexity of medical texts. Machine Learning (ML) models offer a promising solution by classifying medical abstracts, saving time and resources. Current research has explored unsupervised learning but faces challenges with performance. Our project aims to compare unsupervised and supervised models, enhancing classification accuracy using techniques like LoRA and evaluating their potential in real-world medical applications.

## Implementation:
We implemented three supervised models—ELECTRA, RoBERTa, and BERT—enhanced with LoRA for better parameter efficiency and performance. We created a consistent training, testing, and validation split by converting it to Parquet format and uploading it to HuggingFace for better portability. Tokenization and batch processing optimized model training, while validation metrics and loss curves guided the fine-tuning.

## Set Up:

### Prerequisites
Ensure the necessary libraries are installed by running pip install -r requirements.txt to create the environment.

## Running the Experiments

### Basic Models Without LoRA
To run the three baseline models without LoRA, run the following commands:

* BERT without LoRA:
python bert_nolora.py

* RoBERTa without LoRA:
python roberta_nolora.py

* ELECTRA without LoRA:
python electra_nolora.py

### Models With LoRA
To run the models with LoRA run the following python files. Results and metrics will be saved in the respective directories.

* BERT with LoRA:
python bert.py

* RoBERTa with LoRA:
python roberta.py

* ELECTRA with LoRA:
python electra.py

### Ablation Studies
For the ablation studies the rank and alpha values in each script (bert.py, roberta.py, electra.py) are manually adjusted. Start with rank 16 and alpha 32, and halve these values in subsequent runs until reaching rank 4 and alpha 8.

### Hyperparameter Tuning
To perform hyperparameter tuning on the ELECTRA model, run the following:

python tune_electra.py

## Results and Observations:
Our experiments demonstrated that integrating LoRA improved model performance across all three LLMs compared to unsupervised benchmarks. RoBERTa + LoRA emerged as the top performer, surpassing the unsupervised model by significant margins in terms of F1-score. However, challenges like class imbalance in training data and overfitting were observed, suggesting the need for further research and data augmentation strategies.

## Conclusion:
The project highlights the potential of supervised LLMs enhanced with LoRA for medical abstract classification. Future studies could explore advanced techniques like QLoRA to further optimize performance. Ethical considerations include privacy preservation and model validation before clinical deployment. Scaling the experiment to larger datasets could validate its efficacy for broader medical applications.

## Disclaimer:
Experiments were conducted using HTCondor for job scheduling, and the code was pushed at a later stage for analysis and reporting. This project is a prototype and should not be used for clinical or diagnostic purposes without thorough validation and regulatory approval. Ethical considerations, including privacy preservation and model reliability, must be addressed before deployment in medical settings.

## License
This README is intended to provide a detailed and clear description of the set up and process of execution for each experimentation that users can follow.
