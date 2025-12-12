# Analysis of Experimental Results

## 1. Introduction
This section presents a rigorous analysis of the experimental results obtained from training and evaluating multiple news classification models. The primary objective was to quantify the performance benefits of leveraging a Local Large Language Model (DistilBERT) compared to traditional classical machine learning baselines and a generalized Cloud LLM.

## 2. Quantitative Performance Comparison
We evaluated the models on a held-out test set (20% of the corpus). The key performance metrics are summarized below:

### A. Full Test Set Evaluation (N=410)
| Model | Accuracy | F1 Score | Key Characteristic |
|---|---|---|---|
| **Logistic Regression** | 78.5% | 0.78 | Baseline (Linear, Interpretable) |
| **Decision Tree** | 69.8% | 0.70 | Non-linear, Prone to Overfitting |
| **Local LLM (DistilBERT)** | **83.6%** | **0.84** | Fine-tuned, Context-Aware |

![Model Comparison](model_comparison.png)

### B. Cloud LLM Evaluation (Sample N=10)
Due to API rate limits, the **Gemini Flash** model was evaluated on a representative random sample.
*   **Accuracy**: 40.0%
*   **Observation**: The Few-Shot Cloud LLM largely failed to distinguish "Fake" from "Real" news in this specific domain without fine-tuning, exhibiting a strong bias towards classifying articles as "Real" (predicting "Real" 70% of the time). This highlights that general-purpose reasoning is not a substitute for domain-specific fine-tuning when the "fake" cues are subtle stylistic markers rather than obvious absurdities.

## 3. Statistical Significance (McNemar’s Test)
To verify that the performance gap between the Local LLM and Logistic Regression was statistically significant, we conducted **McNemar’s Test**.

**Contingency Table (Local LLM vs Logistic Regression):**
| | Local LLM Correct | Local LLM Wrong |
|---|---|---|
| **Logistic Regression Correct** | 289 | 33 |
| **Logistic Regression Wrong** | 54 | 34 |

*   **P-Value**: **0.0314** ($p < 0.05$)
*   **Conclusion**: There is a **statistically significant difference** in predictive performance, confirming that the Local LLM is the superior model for this task.

## 4. Error Analysis & Confusion Matrices
We visualized the confusion matrices to understand the specific error patterns of the fully trained models.

![Confusion Matrices](confusion_matrices.png)

**Model-Specific Analysis:**
*   **Decision Tree**: exhibited the highest variance, likely overfitting to specific keywords in the training set that did not generalize (e.g., exact phrases like "breaking news").
*   **Logistic Regression**: Performed respectably but struggled with nuanced articles (False Negatives) where the difference between Real and Fake relied on tone rather than vocabulary.
*   **Local LLM**: Reduced the False Negative rate significantly. By using pre-trained embeddings, it correctly identified "Fake" news that mimicked the vocabulary of valid news but lacked the semantic coherence or verified source patterns.

## 5. Training Dynamics (Local LLM)
The Local LLM was trained for up to 20 epochs with early stopping enabled.

![Learning Curve](learning_curve.png)

**Analysis**:
*   **Efficiency**: The model reached peak validation performance efficiently at **Epoch 2**.
*   **Overfitting Prevention**: The Early Stopping mechanism successfully halted training at Epoch 5 when validation loss began to degrade, ensuring the final deployed model was the most generalizable version.
