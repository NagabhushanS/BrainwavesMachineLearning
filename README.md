
![alt text](https://github.com/NagabhushanS/BrainwavesMachineLearning/blob/master/bw.jpg)

# Brainwaves Machine Learning Hackathon 2019
Source code of 1st Place Solution in Brainwaves Machine Learning Hackathon 2019.

# Problem Statement
## Information Extraction from ISDA Legal Documents
ISDA  Documents with 40 clauses and about 200 fields that are to be extracted per document using Machine Learning Techniques. Whole document comprises of up to 40 clauses with up to 250 fields, however for hackathon we are considering 8 clauses which hold 29 essential fields. The objective is to extract the clauses using Machine Learning.

# Approach
## Machine Learning Task
TASK: Automate the process of extraction of key clauses from ISDA documents.

TRAINING DATA: The XML Documents obtained from the OCR of scanned ISDA pdf documents.

INPUT FEATURES and OUTPUT FIELDS: XML text string, 29 Clauses of ISDA documents.

MOTIVATION: To value human time and automate the clause extraction process.

## Project Schematic
The following is a schematic of the architecture:
![alt text](https://github.com/NagabhushanS/BrainwavesMachineLearning/blob/master/architecture.png)

## Exploratory Data Analysis
Each clause may have one or more entries assigned to it. 

The manual clause extraction process requires us to carefuly study the document and requires a great deal of domain knowledge.

My objective is to build a solution which could automate this process.

## XML Feature Engineering
The XML documents are the input features used to train our ML models.

To employ machine learning successfully, we need to clean the data and transform into usable features with the help of Tfidf Transformation

## Machine Learning Models
Once the xml texts and different types of clause fields are preprocessed, we need to train our following models:
  1. Single label LightGBM model for 17 single label clause fields.
  
  2. Multilabel label LightGBM model for 7 multilabel clause fields.
  
Both the types of model are parameter tuned using a RandomizedSearch (from sklearn) instead of a GridSearch.

The trained model are stored (pickled) so that they can be easily be deployed.

## Training Time and Time Per Prediction
The total training time depends on the amount of data. For 249 training documents, the time was around 1 hour.

The averaged time per prediction turned out to be 0.01 sec.

As new training instances are added the model need not be trained on the entire dataset again, but with a warm state need to be trained only on new data added.

## Deployment Strategy and Practical Considerations
A machine learning model once trained can be saved or pickled on the server.

Using a framework like Flask, a post request containing fields as the clauses can be used by the server application to make a prediction on the model after loading the model into memory.

This server can act as an isolated ML server communicating with other types of more general purpose server languages like NodeJS, who delegate prediction requests to the Flask server.


## Dependencies
1. Pandas
2. Numpy
3. NLTK
4. Scikit-Learn
5. LightGBM

## Code Architecture
1. eda.ipynb performs exploratory data analysis.
2. xmlFeaturesEngineering.ipynb generates xml features and performs tokenization and removal of stopwords.
3. trainModel.ipynb trains LightGBM models using tfidf features obtained from the xml texts. The LightGBM models are tuned using RandomizedSearch.

