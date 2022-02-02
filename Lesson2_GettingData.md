---
title: "Getting Data"
---

# Learning Outcomes

- [ ] Access data from various locations
- [ ] Understand what 'big data' is and what implications it has on your models
- [ ] Evaluate the ethics of data collection and use in various applications
- [ ] Understand that a large portion of Data Science involves dealing with the data itself, and how the data collection/storage/memory allocation affects what you are able to do with it.
 
# Data

The first step to build a model is to have data! Data can come from different sources. In the next few sections we provide some examples. 

## Sample Datasets: Experimental Data

Electroencephalogram (EEG) Data
- An EEG is a test that detects electrical activity in the brain. 
- Typically, EEG has a great temporal resolution and the test can be running for few hours at the time. 
- EEG data are used widely to identify different neurological and psychological diseases in patients. 

A free source of EEG data can be found: [https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html](https://sccn.ucsd.edu/~arno/fam2data/publicly_available_EEG_data.html)

## Sample Datasets: Non-Treatment Survey Data 

Explore the 2016 Canadian Census Data, as well as plans for the 2021 Census for free at [https://www12.statcan.gc.ca/census-recensement/index-eng.cfm](https://www12.statcan.gc.ca/census-recensement/index-eng.cfm)

Sampled Survey data can be found all over the web. It's always important to read through the survey questions and sampling procedures to check for bias. 
- The World Values Survey is seen as a credible source of survey data.
- Access the Canadian World Values Survey responses at: [https://www.canada.ca/en/immigration-refugees-citizenship/corporate/reports-statistics/research/world-values-survey-canada-immigrant-native-born-respondent-comparisons.html](https://www.canada.ca/en/immigration-refugees-citizenship/corporate/reports-statistics/research/world-values-survey-canada-immigrant-native-born-respondent-comparisons.html) 

## Sample Datasets: Observational Data

- Remote sensing instruments, such as satellites, radars, and lidars provide valuable observation data. 
    - For example, the International Network for the Detection of Atmospheric Composition Change (NDACC) has more than 70 ground-based remote-sensing research stations most of its data is freely attainable at: [https://www.ndaccdemo.org/data](https://www.ndaccdemo.org/data)

- The Varieties of Democracy (V-Dem) Dataset is the gold standard for social scientists, looking for state-level information about repression, censorship, economic growth, social change, and more. 
    - The raw data are updated annually and available at:
    [https://www.v-dem.net/en/data/data/v-dem-dataset/](https://www.v-dem.net/en/data/data/v-dem-dataset/)
    - One can also explore V-Dem data online on the site's drag-and-drop platform:   [https://www.v-dem.net/en/online-graphing/](https://www.v-dem.net/en/online-graphing/)

## Sample Datasets: Archival Data

- The Spitzer Heritage Archive includes mid-infrared photometry and spectroscopy data.
    - One can access this data at: [https://irsa.ipac.caltech.edu//about.html](https://irsa.ipac.caltech.edu//about.html)

- One can view images of original Emily Dickenson manuscripts, compiled from various rare book rooms at: [https://www.edickinson.org/about](https://www.edickinson.org/about)

- The Arolsen Archive in Bad Arolsen, Germany, has digitized 26 million pages of original materials from WWII and the Holocaust, as well as post-war ship manifests, displaced persons camps records, and International Red Cross refugee interviews. 
    - One can access this data at: [https://arolsen-archives.org/en/](https://arolsen-archives.org/en/)
    
- The UCI Machine Learning Repository is a collection of databases and data generators, well-suited for machine learning projects. 
    - The archive was created in 1987 at UC Irvine. 
    - The database has become one of the major resources for students and researchers in the machine learning community and has been cited more than 1000 times. https://archive.ics.uci.edu/ml/datasets.php

- Kaggle, which is a subsidiary of Google LLC, is an online community for machine learning "lovers". It offers public data platforms as well as machine learning competitions: https://www.kaggle.com/gpreda/covid-world-vaccination-progress

- Plus, individual academics and labs love to share their data on personal and institutional repositories (the latter are sometimes called *Dataverses*)

## Sample Datasets: Simulated Data

There are lots of ways to simulate your own data in ML! For example:
- Synthea, which is a synthetic medical data generator 
    - Learn more at [https://synthetichealth.github.io/synthea/](https://synthetichealth.github.io/synthea/)
- `numpy.random`, a random number generator package for Python



# Big Data

<img src="figs/big_data.png" width=280>

Sometimes data sets become too large or complex to be captured and stored within the capability of traditional sofware, let alone analyzed, shared, and transformed. 

Two tools for handling big data are: *Cloud Computing* and *Deep Learning*. 
- Cloud Computing refers to on-demand cloud-based computing systems, available to many users over the internet 
    - Think about programs and databases that free up space on your computer, like Google Cloud, Amazon Web Services, and Microsoft Azure.
- Deep Learning (aka Deep Neural Learning or Deep Neural Networks) refers to a type of machine learning that uses artificial intelligence to imitate the human brain in the study of big data. 
	- The advantage of deep learning is that it can learn one case at a time - there's no need to have all the data in one place!

More information on big data can be found:

[https://en.wikipedia.org/wiki/Big_data#Technologies](https://en.wikipedia.org/wiki/Big_data#Technologies)

# Where Can You Put Data?

## On your own hard drive (backed up, of course)

- Back up your data!
    - External hard drive, if possible
    - Cloud drives
- Give it a meaningful name
- Keep it organized in folders 
- Specify the raw (unmodified) data and data that have been cleaned/modified in the file name and folder structure

Note: This only works for data that are small enough to fit on your hard drive!

## In the Cloud

- Most services offer automatic back-ups and version control
- With paid accounts, you can often store larger files
- Can be used as a backup option
    - You can zip your data to save space, if that's an issue
        - Zipping is essentially compressing lots of files or large files into a small folder (like you zip up your favourite sweater)
            - It's easy to share with others and open - just double-click and your computer will display the contents of the zipped folder. 

<img src="figs/cloud.png" width=280>

## Amazon Web Services (AWS)

AWS is a cloud storage solution that allows for both:
- Automation of tasks (e.g. downloading new data as it becomes available) and 
- Cloud based computation.

AWS uses a tool called SageMaker, which helps you to use somebody else's computational power to fit your machine learning models. It's a wonderful, expensive system.



# How Do I Get Data Into My Model? 

If the data are on your computer, then it's generally not hard to fit your model with those data. 

There will almost certainly be a few cleaning/pre-processing steps and you should always *always* **always** plot your data first to learn about the contents of different variables
- You can do this in Python
- You can also use a drag-and-drop program like Tableau
    - Tableau is **free** for educators/students with a valid university email address
        - You can download the desktop app or use the cloud-based version

After you plot your raw data to map out *what* you have, you can assign variables and include them in your model. 
- This is a pretty straightforward process that we will teach in this bootcamp.



# Lesson Two Wrap-Up

In this lesson, we:
- [x] Access data from various locations
- [x] Understand what 'big data' is and what implications it has on your models
- [x] Understand that a large portion of Data Science involves dealing with the data iteself, and how the data collection/storage/memory allocation affects what you are able to do with it.


# See you in Unit 1, Lesson 3












