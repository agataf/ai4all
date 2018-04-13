# Background
The Fragile Families Challenge is a large-scale interdisciplinary research initiative to collect and analyze data in order to yield insights that can improve the lives of disadvantaged children.  Researchers have used this data to develop models that predict key attributes affecting disadvantaged children and to suggest new policies to improve child outcomes. In this project, you will use data collected as a part of the Fragile Families Challenge to uncover factors that influence young people’s academic performance, confidence and grit, and psychological well-being. You will generate scientific questions and perform data exploration, feature selection, and machine learning to evaluate your hypotheses. You will explore alternative explanations for your results and work closely with Princeton social scientists to refine your hypotheses. You will also work together to design policy proposals based on your findings that would help provide services and programs to facilitate children’s success.

More information:

  http://www.fragilefamilieschallenge.org/
  
Watch the "getting started" video:

  https://www.youtube.com/watch?v=HrYPtdXeSaM&feature=youtu.be
  
See the "getting started" slides here:

  https://github.com/fragilefamilieschallenge/slides/blob/master/ffchallenge_getting_started_cos424.pdf
  
# Data
We use the Fragile Families Challenge data. The data has been collected since children's birth at 7 time points: birth, year 1, 3, 5, 9 and 15. The goal of the challenge is to predict 6 outcomes (GPA, grit, eviction, job training, 2 more) at age 15 based on the variables from earlier time points. There are more than those 6 variables that we could get access to for age 15, if they'll be easier.

The train and test splits are done for us. The training set consists of 12,000 variables across 3,200 families. The test set consists of 6 variables across ~2,000 families. Our goal is to predict those six variables for the remaining ~1,000 families.

Each variable has a dictionary of features associated with it, such as: source (constructed/weight/id number/...), respondent (father/mother/teacher), umbrella category (parental relationship, health and health behavior,...) and others. You can view variables by their features here: http://browse.fragilefamiliesmetadata.org/variables.

Starting this year, there is also an API available which allow us to subselect only the variables with particular features. There are 3 functions in this API:

`select(varName, fieldName)`
    Returns metadata for variable varName.
    (Optionally, returns only the field specified by fieldName.)

`filter(*fieldNames)`
    Return a list of variables where fieldName matches the provided value.

`search(query, fieldName)`
    Return a list of variables where query is found in fieldName.
 
 To access the api, you need the `ff.py` file in your directory, then `import ff` and call the abovementioned functions as `ff.select(...)`
 
 Examples of the API use can be seen in `preprocess.ipynb`

# Setup:
- clone this directory 
  `cd ai4all_dir`

  `git clone https://github.com/agataf/ai4all`
- you should technically request data from the SOC department, but for the three of us I'm putting it up here. Please don't share this with anyone. 
  * https://drive.google.com/drive/folders/1DYnjfqIxZCrAYc7juV0UplvZlwTNDxJ8?usp=sharing
- download the data to a directory called ai4all_data next to ai4all

  `cd ai4all_dir`

  `cp path_to_data/ai4all_data .`

- make sure you have jupyter notebook downloaded

  `python -m pip install --upgrade pip`

  `python -m pip install jupyter`

- I'm assuming Python2.7 - let me know if you want to use Python3
- to run jupyter notebook:

  `cd ai4all_dir/ai4all`

  `jupyter notebook`
