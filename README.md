# JoyRide: Vehicle Review Sentiment Analysis

A comprehensive Natural Language Processing project that analyzes customer reviews of vehicles to extract insights about consumer sentiments, preferences, and common themes. This project demonstrates advanced NLP techniques including sentiment classification and topic modeling.

## Overview

JoyRide leverages machine learning and NLP methodologies to process and analyze thousands of vehicle reviews. The project implements two main analytical approaches:

- **Sentiment Analysis**: Classifying reviews as positive, negative, or neutral using Support Vector Machine (SVM)
- **Topic Modeling**: Discovering common themes in reviews using Latent Dirichlet Allocation (LDA)

## Features

- **Text Preprocessing Pipeline**: Complete text cleaning workflow including punctuation removal, tokenization, stopword removal, stemming, and lemmatization
- **TF-IDF Feature Extraction**: Converts text data into numerical features for machine learning models
- **SVM Sentiment Classification**: Linear kernel SVM for predicting review sentiment
- **LDA Topic Modeling**: Identifies 11 distinct topics across vehicle reviews
- **Data Visualization**: Visual analysis of sentiment trends and topic distributions
- **Comprehensive Documentation**: Well-documented Jupyter notebook with markdown explanations

## Dataset

The project includes multiple datasets located in the `Dataset/` folder:

| Dataset                          | Records | Description                                                 |
| -------------------------------- | ------- | ----------------------------------------------------------- |
| `vehicle_reviews_full.csv`       | 10,678  | Full dataset used for final SVM model training              |
| `vehicle_reviews_sample.csv`     | 5,540   | Sample dataset for initial exploration and faster iteration |
| `vehicle_reviews_sample_svm.csv` | 5,540   | Alternative sample dataset                                  |
| `vehicle_reviews_train.csv`      | 2,870   | Pre-split training set for alternative workflows            |
| `vehicle_reviews_test.csv`       | 2,273   | Pre-split test set for alternative workflows                |

### Dataset Schema

Each dataset contains the following columns:

- **Company**: Vehicle manufacturer (e.g., Acura, BMW, Honda)
- **Model**: Specific vehicle model
- **Year**: Manufacturing year of the vehicle
- **Reviewer**: Name or identifier of the reviewer
- **Date**: Review submission date
- **Title**: Brief title of the review
- **Rating**: Customer rating (categorical: pos, neg, neu or numerical: 1-5)
- **Review**: Full text of the customer review

## Models and Performance

### Support Vector Machine (SVM) - Sentiment Classification

**Configuration**:

- Kernel: Linear
- Feature Extraction: TF-IDF (min_df=5, max_df=0.8)
- Train-Test Split: 80:20
- Random State: 42

**Performance Metrics**:

- **Overall Accuracy**: 89%
- **Positive Sentiment**: Precision 0.91, Recall 0.98, F1-Score 0.95
- **Negative Sentiment**: Precision 0.55, Recall 0.53, F1-Score 0.54
- **Neutral Sentiment**: Precision 1.00, Recall 0.03, F1-Score 0.05

**Analysis**: The model performs excellently on positive reviews but shows moderate performance on negative reviews and poor recall on neutral reviews, indicating class imbalance that could be addressed with resampling techniques or class weights.

### Latent Dirichlet Allocation (LDA) - Topic Modeling

**Configuration**:

- Number of Topics: 11
- Passes: 50 (with early stopping)
- Early Stopping: Based on coherence score (patience=5)
- Text Input: Lemmatized reviews

**Topics Discovered**:

1. Reliability
2. Comfort
3. Transmission
4. Fuel Efficiency
5. Speed
6. Mileage
7. Safety
8. Accessories
9. Price
10. Attractiveness
11. Color

## Installation

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Required Packages

Install all required dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib nltk gensim seaborn
```

### NLTK Data Downloads

After installing the packages, download required NLTK data:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Usage

### Clone the Repository

```bash
git clone https://github.com/devYRPauli/NLP-JoyRide.git
cd NLP-JoyRide
```

### Run the Jupyter Notebook

1. Start Jupyter Notebook:

   ```bash
   jupyter notebook JoyRide_NLP.ipynb
   ```

2. Execute cells sequentially to:
   - Load and explore the dataset
   - Preprocess text data
   - Perform topic modeling with LDA
   - Train SVM classifier for sentiment analysis
   - Visualize results and insights

### Using Different Datasets

The notebook is configured to use:

- `vehicle_reviews_sample.csv` for initial exploration
- `vehicle_reviews_full.csv` for final SVM training

To use the pre-split train/test datasets, modify the data loading cells accordingly.

## Project Structure

```
NLP-JoyRide/
│
├── Dataset/
│   ├── vehicle_reviews_full.csv           # Full dataset (10,678 reviews)
│   ├── vehicle_reviews_sample.csv         # Sample dataset (5,540 reviews)
│   ├── vehicle_reviews_sample_svm.csv     # Alternative sample dataset
│   ├── vehicle_reviews_train.csv          # Pre-split training set
│   └── vehicle_reviews_test.csv           # Pre-split test set
│
├── JoyRide_NLP.ipynb                      # Main Jupyter notebook
├── README.md                               # Project documentation
└── LICENSE.md                              # License information

```

## Text Preprocessing Pipeline

The notebook implements a comprehensive preprocessing pipeline:

1. **Punctuation Removal**: Strips special characters and punctuation marks
2. **Lowercasing**: Converts all text to lowercase for uniformity
3. **Tokenization**: Splits text into individual words using NLTK
4. **Stopword Removal**: Removes common English stopwords
5. **Stemming**: Reduces words to root form using Porter Stemmer
6. **Lemmatization**: Converts words to base dictionary form using WordNet Lemmatizer

## Visualization

The notebook includes various visualizations:

- Rating distribution analysis
- Sentiment distribution across vehicle models
- Topic distribution across reviews
- Average ratings by topic over time
- Word clouds for different topics

## Team Members

- Abhinav Ayral
- Ansh Khatri
- Ayushma Joshi
- Ria Mahajan
- Yash Raj Pandey

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

Please ensure your code follows the existing style and includes appropriate documentation.

## Future Improvements

Potential enhancements for the project:

- Address class imbalance in sentiment classification using SMOTE or class weights
- Implement additional models (Random Forest, Naive Bayes, Neural Networks)
- Hyperparameter tuning for improved model performance
- Real-time sentiment analysis from web-scraped reviews
- Interactive dashboard for visualizing insights
- Multi-language support for international reviews

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Dataset sourced from vehicle review platforms
- NLTK and Gensim communities for excellent NLP tools
- Scikit-learn for machine learning implementations
