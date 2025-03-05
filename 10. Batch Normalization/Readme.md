### Batch Normalization
- **Concept**: A technique to standardize the inputs to a layer in a deep learning neural network.
  - **Mean**: The arithmetic average of a dataset.
  - **Standard Deviation**: Measures the dispersion of values in a dataset.
  - **Normalization**: Converts data into a standard range (e.g., -1 to +1 or 0 to 1).
- **Benefits**:
  - Makes neural networks more stable.
  - Enables higher learning rates.
  - Reduces overfitting.
- **Momentum**: Controls how much of the statistics from the previous mini-batch are included in the update.

### Deep Learning Terminology
- **Confusion Matrix**: An NxN table summarizing the performance of a classification model.
  - **Metrics**:
    - Accuracy: (TP + TN) / Total
    - Misclassification Rate: (FP + FN) / Total
    - True Positive Rate (Recall): TP / Actual Yes
    - False Positive Rate: FP / Actual No
    - Precision: TP / Predicted Yes
    - Prevalence: Actual Yes / Total
- **Convergence**: A state during training where the loss changes very little between iterations.
- **Classification Types**:
  - Binary Classification
  - Multiclass Classification
  - Multilabel Classification
  - Imbalanced Classification
- **Downsampling**: Reducing the amount of information in a feature to train a model more efficiently.
  - Helps balance training on majority and minority classes in imbalanced datasets.

---

## Key Takeaways
- Batch normalization is a powerful technique to stabilize and improve the performance of neural networks.
- Confusion matrices provide a detailed breakdown of a model's classification performance.
- Downsampling is useful for handling imbalanced datasets and improving model efficiency.
