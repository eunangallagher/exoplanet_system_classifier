## Milestone 2
I have opened, cleaned, preprocessed and split the dataset into x/y_train and x/y_test 
datasets. Then i used a dummy classifier as my baseline model to determine how much better
trained models will perform.
#
## Dataset choice and target variable setup.
Source: The Planetary Systems Composite Parameters (PSCompPars) table from the official NASA Exoplanet Archive.
Inupt features: 6 physical stellar properties (`number of stars: sy_snum`, `effective temperature: st_teff`,
 `stellar radius: st_rad`, `stellar mass: st_mass`, `stellar metallicity: st_met`, `surface gravity: st_logg`).
Target variable: I grouped the dataset into each unique stellar system, then i counted the number of planets in       each system and set the target variable = 0 if there is one planet in the system and set the target variable = 1 if there is more than 1 planet. The target distribution is heavily skewed to systems with 1 planet. 

