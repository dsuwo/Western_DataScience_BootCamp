---
title: "Cleaning Data"
---

# Learning Outcomes

- [ ] Recognize common issues with data, related to data cleaning
- [ ] Become familiar with various data cleaning techniques 
- [ ] Understand how data cleaning processes fit into the overall process of data collection, organization, and analysis
- [ ] Know what a codebook is and why researchers create this document to accompany original datasets

Video version of this lesson:

<a href="https://www.youtube.com/watch?v=VlJ-cOsoSP4"><img src = "https://i.ytimg.com/vi/VlJ-cOsoSP4/hqdefault.jpg"></a>

# Data

One of the most challenging and time consuming parts of developing machine learning algorithms is data pre-processing. Here, we talk about this in more details.   

# How Does Data Become Unclean? 

This can happen because of human error in user-input data. 
For example, human error can appear as: 
- Differences in capitalization or notation (e.g., Male, male, M)
- Differences in naming (e.g., Kiev, Kyiv) 
- Differences in data entry (e.g., 1998, 98, '98)
- Notation for unknown (e.g., unknown, N/A, 0, -99)
- Double-counting (e.g., when an observation appears more than once)
- Typos because [spelling can be hard](https://pudding.cool/2019/02/gyllenhaal/)
    - Notice how the visualizations show us what the data contains!

This can also happen with data collected from electroics/instruments
For example:
- General anomalies in our instruments 'behaviour' 
- Some instruments have background noise that interferes with the study. 
    - All instruments have background noise. 
    - Sometimes the background noise is just a constant value and we can easily subtract this value. 
    - Many other times this background noise value varies and we need to find a way to deal with it! If not our analysis will be incorrect. 
- There are some 'Not-A-Number' values (e.g., NAN/nan/NaN) in the file because the instrument was not operating at one time, but an observer continued to use this notation, even after it was no longer appropriate.

# What is Data Cleaning?

When researchers clean data, they seek:
- Uniformity
- Reliability 
- Consistency
- Accuracy 
- Completeness 
- Validity

These attributes may sound similar, but in practice they operate in different ways. 
- For example, a variable could be complete (contains all values) and consistent (those values are all recorded in the same/ correct format), but still be inaccurate or unreliable, if the values entered do not reflect the real world. 
    - As a result, in data cleaning, researchers perform techniques related to **compliancy/precision** as well as **substantive correctness**. 

# Before You Clean

Back up your data!

The cleaning process will result in a modified data set. If it's small enough to be stored on Github, the original version could theoretically be recovered. Otherwise, you might accidentally delete important information.

It is common practice to have a folder for raw data and a folder for cleaned data (assuming you can fit the data on your computer twice). 
- The raw data should be **read-only** so that no modifications can be made.
    - It's useful to store the original data dictionary and a source attribution/reference in the same folder.
- The cleaned data should be created by a script (e.g., Python code) so that you have documentation about what changed and why.
    - This script should be in the same folder as the cleaned data.

# Techniques for Data Cleaning

Data Cleaning has four steps:
- Inspection
- Cleaning
- Verification 
- Reporting Changes Made and/or An Update About the Quality of the Dataset 

Each of the steps are labour-intensive. After all, we cannot perform the task of *cleaning* the data until we know *what needs to be cleaned*. 

The following techniques for data cleaning can be done both manually and computationally, but there is often a need to spot-check by hand, even after computational cleaning:

- **You must remove irrelevant values**
    - This is any kind of 'useless' and invalid information, such as a numerical entry for a categorical variable
- **You must get rid of duplicate entries** 
    - By definition, we want our observations to be independent of one another 
- **You must address typos and inaccurate information**
   - For example, in a study of divorce, a respondent who is recorded as 10 years old would fall outside the range constraints of your variable (and, if left in the dataset, could skew your results!)
   - Likewise, a person cannot become fired for their job in September 1994 but begin the position in November 1994. The dates must *make sense*.
- **You must convert data types** 
- **You must address missing values** 
- **You must ensure that entries match your predetermined 'expression patterns'**
    - For example, all dates as dd/mm/yy vs. dd/mm/yyyy vs. mm/dd/yy

# Cleaning for Compliancy/ Precision

A variable represents data that is stored. 

When saving data in computer, data can be stored in different variable 'types'. 

These types are:
- Numbers 
    - An Integer (commonly called an int) is a number without a decimal point
    - A Floating-point Number (commonly called a float, or double-precision, or just a double) is a number with a decimal place 
        - Floats are useful when we need more precision 
- Strings, which is a type of variable that includes characters, such as unstructured text as well as category names.  
- Booleans, which is a type of variable that presents two possible values (True or False).

The goal is complete consistency for each variable. 
- For example, height to always be measured in centimeters with a decimal point (e.g. 177.0 rather than 177), indicating that it is a float.

But the reality is that humans are inconsistent! 
- A yes/no question might be recorded as:
    - yes/no
    - Yes/ No
    - YES/ NO
    - 1/0
    - True/False
    - true/false
    - TRUE/FALSE
    
Before we do *any* analysis, we have to revise our dataset so it is all recorded in a consistent way. 
- Otherwise, the models we use will inevitably give us error messages.

# Cleaning for Substantive Correctness 

A dataset should make sense! 

For example, if we are looking at the average day time temeperature for London, Ontario, during winter, and all of the recorded values are between 75 to 85 Fahrenheit, we should get suspicious. 

To do the 'sanity-check', we can:
- Look through the raw data,
- Hire an RA to spot-check our dataset (like a fact checker at a newspaper!)
- Visualize summary statistics for various variables with Python or Tableau 
    - Major errors will pop out at us 
        - This is one place where substantive expertise matters!
        
# Machine Learning for Missing Data

ML is uniquely positioned to handle issues related to missing data, as these methods can help to reconstruct what the original distribution might have looked like. 
- For example, statistical and machine learning techniques can solve missing data problems through **deletion** and **imputation**.

Deletion can be carried out through:
- Listwise deletion (also called complete case analysis) 
    - Through listwise deletion, a researcher or their algorithm would only analyze cases with available data on every variable
- Pairwise deletion (also referred to as available case analysis)
    - With pairwise deletion, an analysis would focus only on observations where the variable of interest is present
    
Alternatively, researchers can respond to missing data through imputation
- Imputation is the act of creating new data to account for missing entries. 
    - This is frequently done through:
        - Mean-based substitution, or 
        - The creation of so-called ‘fake datasets’ based on predetermined parameters.
        - A regression of the feature with missing values against the other features (independently of the variable of interest) to find the most reasonable imputation value.

# Codebooks, Because Data are Meant to be Shared 

If we do decide to collect our own research, it is necessary to create a **codebook**.

Codebooks are what makes science truly replicable! 
Good codebooks are intentionally thorough: 
- A codebook will discuss **every single variable** that was collected, how it was measured, and how to replicate a study *step-by-step*.
- It should also explain how you dealt with every problem and outlier you might run in to.
- It also opens a researcher to challenge, as other scholars can evaluate (and falsify) the operationalization of important variables. 

Any (good) dataset will have a codebook. This will often be published in tandem with any study that is published, in which the researcher collected and operationalized original data. 

Generally, this takes the form of a docx file that contains a table with two columns - one for the variable name and one for the description. Outside the table will be extra information, such as the reference for the paper that the data were published with and any other acknowledgements/attributions. 

Some examples can be found here: [https://www.sheffield.ac.uk/mash/statistics/datasets](https://www.sheffield.ac.uk/mash/statistics/datasets)

# How to Actually Clean Your Data

If it's a small enough data set and the problems are relatively basic, this is the only place where we will advocate for Excel.
- For instance, if you open the data and see an M instead of an M, go ahead and change it (and add this change to the codebook!).
    - Doing this sort of change is actually very tedious in Python

If there are a lot of things to be changed or a whole column to be changed, use a script (e.g. Python or R or Bash scripting).
- For instance, if your data are recorded as `Name (Age)`, e.g. Johanna (25) and Marty (17), you could write a *regular expression* (aka regex) to convert this into two columns - one for name and one for age.

Script files provide a paper trail for all of the changes you made to the data. You know exactly how everything was changed because you can read the instructions yourself! They can also be re-run to ensure that all of the cleaning steps always happen in the same order.


# Lesson 1 Wrap-Up

In this lesson, we:
- [x] Recognized common issues with data, related to data cleaning
- [x] Became familiar with various data cleaning techniques 
- [x] Began to understand how data cleaning processes fit into the overall process of data collection, organization, and analysis
- [x] Learned what a codebook is and why researchers create this document to accompany original datasets



# See you in Unit 2, Lesson 2: Intro To Pandas!
