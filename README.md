# Unsupervised-and-Supervised-LLMs-with-LoRA-for-Medical-Abstract-Text-Classification

In the medical field, diagnosing diseases accurately and early is crucial but challenging due to the complexity of medical texts. Machine Learning (ML) models offer a promising solution by classifying medical abstracts, saving time and resources. Current research has explored unsupervised learning but faces challenges with performance. Our project aims to compare unsupervised and supervised models, enhancing classification accuracy using techniques like LoRA and evaluating their potential in real-world medical applications.

## Implementation:
We implemented three supervised models—ELECTRA, RoBERTa, and BERT—enhanced with LoRA for better parameter efficiency and performance. We created a consistent training, testing, and validation split by converting it to Parquet format and uploading it to HuggingFace for better portability. Tokenization and batch processing optimized model training, while validation metrics and loss curves guided the fine-tuning.

## Results and Observations:
Our experiments demonstrated that integrating LoRA improved model performance across all three LLMs compared to unsupervised benchmarks. RoBERTa + LoRA emerged as the top performer, surpassing the unsupervised model by significant margins in terms of F1-score. However, challenges like class imbalance in training data and overfitting were observed, suggesting the need for further research and data augmentation strategies.

## Conclusion:
The project highlights the potential of supervised LLMs enhanced with LoRA for medical abstract classification. Future studies could explore advanced techniques like QLoRA to further optimize performance. Ethical considerations include privacy preservation and model validation before clinical deployment. Scaling the experiment to larger datasets could validate its efficacy for broader medical applications.
