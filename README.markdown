# Sentiment Analysis of Amazon Fine Food Reviews

## Overview

This project performs sentiment analysis on the Amazon Fine Food Reviews dataset using Python. It employs three different techniques to analyze the sentiment of the reviews:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A lexicon-based, bag-of-words approach for sentiment analysis.
2. **RoBERTa Pretrained Model**: A transformer-based model from HuggingFace for advanced sentiment classification.
3. **HuggingFace Pipeline**: A high-level interface for performing sentiment analysis using pre-trained models.

The notebook processes a subset of the dataset (first 500 reviews) and compares the sentiment scores from these techniques with the user-provided review scores. It also visualizes the results and highlights discrepancies between model predictions and actual ratings.

## Dataset

The dataset used is the **Amazon Fine Food Reviews** dataset, sourced from Kaggle (`snap/amazon-fine-food-reviews`). It contains user reviews for food products, including:

- **Text**: The review text.
- **Score**: The rating (1 to 5 stars) provided by the user.
- Other metadata such as Product ID, User ID, Helpfulness scores, etc.

For computational efficiency, the notebook processes only the first 500 reviews from the dataset.

## Prerequisites

To run the notebook, you need the following dependencies installed:

- Python 3.x
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `kagglehub`
  - `transformers` (for HuggingFace models)
- NLTK data packages:
  - `vader_lexicon`
  - `punkt_tab`
  - `averaged_perceptron_tagger_eng`
  - `maxent_ne_chunker_tab`

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn nltk kagglehub transformers
```

To download the required NLTK data, run the following in the notebook:

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
```

## Notebook Structure

The notebook is organized into the following steps:

1. **Data Import and Setup**:

   - Downloads the Amazon Fine Food Reviews dataset using `kagglehub`.
   - Loads the dataset into a Pandas DataFrame and selects the first 500 reviews for analysis.

2. **Exploratory Data Analysis**:

   - Uses visualization tools (`seaborn` and `matplotlib`) to explore the dataset.
   - Generates a pairplot to compare sentiment scores (VADER and RoBERTa) with the user-provided review scores.

3. **Sentiment Analysis**:

   - **VADER**: Applies the VADER sentiment analyzer to compute positive, negative, and neutral sentiment scores for each review.
   - **RoBERTa**: Uses a pre-trained RoBERTa model from HuggingFace to calculate sentiment scores.
   - **HuggingFace Pipeline**: Leverages the HuggingFace `pipeline` for a simplified sentiment analysis workflow.

4. **Review Examples**:

   - Identifies and displays reviews where the model predictions significantly differ from the user-provided scores (e.g., positive sentiment for 1-star reviews and negative sentiment for 5-star reviews).

## Key Findings

- The pairplot visualization (`sns.pairplot`) shows the relationship between VADER and RoBERTa sentiment scores and the user-provided ratings.
- Examples of discrepancies include:
  - **Positive 1-Star Review**: A review mentioning feeling energized but criticizing the cost-effectiveness, leading to a high positive sentiment score despite the low rating.
  - **Negative 5-Star Review**: A review praising the taste but noting negative consequences (e.g., weight gain), resulting in a high negative sentiment score despite the high rating.

## How to Run

1. Clone or download this repository.
2. Ensure all prerequisites are installed.
3. Open the notebook (`Sentiment_Analysis_of_Amazon_Fine_Food_Reviews_Python.ipynb`) in Jupyter Notebook or Google Colab.
4. Run the cells sequentially to:
   - Download the dataset.
   - Install required NLTK data.
   - Perform sentiment analysis and visualize results.

## Limitations

- The analysis is limited to the first 500 reviews due to computational constraints.
- Sentiment models may misinterpret sarcasm, context, or mixed sentiments, leading to discrepancies with user ratings.
- The dataset path in the notebook assumes a specific file structure (`../kaggle/input/amazon-fine-food-reviews/Reviews.csv`). Adjust the path if necessary based on the downloaded dataset location.

## Future Improvements

- Expand the analysis to include the full dataset for more robust insights.
- Incorporate additional sentiment analysis techniques or models for comparison.
- Add preprocessing steps (e.g., text cleaning, removing stopwords) to improve model performance.
- Explore advanced visualization techniques to better understand sentiment distribution.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The Amazon Fine Food Reviews dataset is provided by SNAP (Stanford Network Analysis Project) via Kaggle.

# Sentiment analysis techniques utilize the `nltk` library for VADER and HuggingFace's `transformers` for RoBERTa and pipeline methods.