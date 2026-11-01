
# Milestone 2 Data audit and report
## Eunan o Gallachoir   
## 23437944
## e.ogallachoir1@universityofgalway.ie
I have opened, cleaned, preprocessed and split the dataset into x/y_train and x/y_test   
datasets. Then i used a dummy classifier as my baseline model to determine how much better  
trained models will perform.
## Module Versions:
* Pandas version: 3.0.6  
* NumPy version: 2.5.4  
* Scikit-learn version: 1.9.1  

## Data Audit
* Raw data shape: (6451, 10)  
* Missing values check:  
pl_name            0  
hostname           0  
default_flag       0  
sy_snum            0  
sy_pnum            0  
st_teff          773  
st_rad           841  
st_mass          877  
st_met          2009  
st_logg         1136  
dtype: int64  

* Cleaned Data  
Clean systems left: 3343  
X_train shape: (2674, 6)  
X_test shape: (669, 6)  
  
* Class counts in training:  
0 (Single); 2204  
1  (Multi); 470  
  
* Class percentages in training:  
0    0.824233  
1    0.175767  
  
##Dummy Classifier Performance:  
* Overall Baseline Accuracy: 82.36%
* Single-planet Class Accuracy: 100.00%
* Multi-planet Class Accuracy: 0.00%
  
* Confusion Matrix:  
    [[551   0]  
    [118   0]]  

True Positives: 0  
False Positives: 0  
False Negatives: 118  
True Negatives: 551  
  
## Dataset choice and target variable setup.  
Source: The Planetary Systems Composite Parameters (PSCompPars) table from the official NASA Exoplanet Archive.  
Inupt features: 6 physical stellar properties (`number of stars: sy_snum`, `effective temperature: st_teff`,
 `stellar radius: st_rad`, `stellar mass: st_mass`, `stellar metallicity: st_met`, `surface gravity: st_logg`).  
Target variable: I grouped the dataset into each unique stellar system, then i counted the number of planets in each system and set the target variable = 0 if there is one planet in the system and set the target variable = 1 if there is more than 1 planet. The target distribution is heavily skewed to systems with 1 planet.   
Train/test split: I chose an 80/20 split to maximise the data my model can train on while keeping enough data to evaluate its performance well. And i stratified by the target variable to retain class proportions in the train and test sets.  

## Assitance Declaration
I used GitHub CoPilot in vscode as an assitant for the project milestone. I used it to help with pandas and scikit-learn method syntax. All code has been checked and verified by me to ensure it works