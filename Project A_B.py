#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[233]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[234]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[235]:


len(df.index)


# c. The number of unique users in the dataset.

# In[236]:


df.nunique()['user_id']


# d. The proportion of users converted.

# In[237]:


# Mean of columns converted
df['converted'].mean()


# e. The number of times the `new_page` and `treatment` don't match.

# In[238]:


Group1 = df.query("group == 'treatment' and landing_page == 'old_page' ")
Group2 = df.query("group == 'control' and landing_page == 'new_page' ")

# To find the number of times the new_page and treatment don't match
number_not_match = len(Group1)+ len(Group2)
number_not_match


# f. Do any of the rows have missing values?

# In[239]:


# Check if there's any null values
df.info()


# In[240]:


df.isnull().sum()


# ##### There is no missing values

# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[241]:


# To create a new dataset and Store it as a new dataframe in df2
df2 = df.copy()
# If the control does not match with old_page or treatment does not match with new_page
df2 = df[((df.group=='control') & (df.landing_page=='old_page')) | ((df.group=='treatment') & (df.landing_page=='new_page')) ]
df2.head()


# In[242]:


# The result have to be zero
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[243]:


# Check how many unique user_ids are in df2
df2.nunique()['user_id']


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[244]:


# Check the repeated user_id in df2
df2[df2.duplicated('user_id')]


# c. What is the row information for the repeat **user_id**? 

# In[245]:


# Check information for the repeat user_id
df2[df2.user_id == 773192]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[246]:


# Remove one of the rows with a duplicate user_id
df2.drop(2893, inplace=True)
df2[df2.user_id == 773192]


# In[247]:


# Check if one of the duplicated values are removed
df2[df2.user_id == 773192]


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[248]:


# Check the probability of an individual converting regardless of the page they receive
df2['converted'].mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[249]:


# Check the probability they converting if that an individual was in the control group
control_group = df2[df2.group == 'control'].converted.mean()
control_group


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[250]:


# Check the probability they converting if that an individual was in the treatment group
treatment_group = df2[df2.group == 'treatment'].converted.mean()
treatment_group


# d. What is the probability that an individual received the new page?

# In[251]:


# Check the probability that an individual receiving the new page
new_page = (df2['landing_page'] == "new_page").mean()
new_page


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Based on the results we can say the probability of individual conversion of the treatment group new page (0.11) is slightly less than the probability of individual conversion of the control group old page(0.12).
# The probability that an individual received the new page is (0.50), not a big difference between the conversion of new and old page. It means the new treatment page does not leads to more conversions and we can't be sure if this is significant or not because the difference is so low.**
# 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# $H_{1}$:  $p_{old}$  <  $p_{new}$

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[252]:


# Check the conversion rate for  𝑝𝑛𝑒𝑤  under the null
cr_pn = df2['converted'].mean() 
cr_pn


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[253]:


# Check the conversion rate for  𝑝𝑜𝑙𝑑  under the null
cr_po = df2['converted'].mean() 
cr_po


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[254]:


# Find the number of individuals in the treatment group
n_new = df2[df2['group'] == 'treatment'].shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[255]:


# Find the number of individuals in the control group
n_old = df2[df2['group'] == 'control'].shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[256]:


# To find samples with probability of new page
new_page_converted = np.random.choice([1,0], size=n_new, p=[cr_pn, (1-cr_pn)])
len(new_page_converted)                                    


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[257]:


# To find samples with probability of old page
old_page_converted = np.random.choice([1,0], size=n_old, p=[cr_po, (1-cr_po)])
len(old_page_converted)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[258]:


# Computing the difference between the new page and the old page 
diff_computed= (df2[df2['group'] == "treatment"]['converted'].mean()) - (df2[df2['group'] == "control"]['converted'].mean())
diff_computed


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[259]:


# Create 10,000 𝑝𝑛𝑒𝑤- 𝑝𝑜𝑙𝑑 values based on the above parts
p_diffs = []

NPC = np.random.binomial(n_new,cr_pn,10000)/n_new # dividing on the size of respective population
OPC = np.random.binomial(n_old,cr_po,10000)/n_old # dividing on the size of respective population
p_diffs = NPC - OPC #Store all 10,000 values
p_diffs


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[260]:


plt.hist(p_diffs)
plt.xlabel('Probability difference');
plt.ylabel('Count');
plt.axvline(diff_computed, color='y');


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[261]:


#Convert p_diffs to numpy array
p_diffs = np.array(p_diffs)
# compare he actual size 'diff_computed' with p_diffs
(diff_computed < p_diffs).mean()


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# ###### The value computed in part j above it called p-value, and this value (0.90) greater than the actuall difference obsreved in ab_data (-0.0015). It means we can't reject the null hypothesis because there is no sufficient evidence between the new and old pages.
# 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[262]:


import statsmodels.api as sm
# Number of conversions for the old page
convert_old = df2.query(" landing_page == 'old_page' & converted == 1").shape[0]
# Number of conversions for the new page
convert_new = df2.query(" landing_page == 'new_page' & converted == 1").shape[0]
# Number of individuals who received old page
n_old = df2[df2['group'] == 'control'].shape[0]
# Number of individuals who received new page
n_new = df2[df2['group'] == 'treatment'].shape[0]
convert_old, convert_new, n_old, n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[263]:


# compute the z_score and p_value
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Yes I agree with parts j and k, because the value of z_score is (1.31) lower than the critical value, It means we accept the null hypotheses and reject alternative hypotheses.**

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# 
# **I'll use (Logistic regression) because it's a categorical data.**

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[264]:


# Create intercept column
df2['intercept'] = 1
# create a dummy variable column
df2['ab_page'] = pd.get_dummies(df2['group'])['treatment']
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[265]:


regression_model = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
fit_model = regression_model.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[266]:


from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)
fit_model.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **The p-value associated with ab_page is (0.190), not look the value we found in Part II because we used only one tailed test, but here in this part we used two tailed test in the logit regression, also we still accept the null hypothesis and reject alternative hypotheses beause of the p-value is still high**

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Yes, It's a good idea to consider other factors to add into the regression model, and for sure there's a disadvantages to adding additional terms into the regression model, needs to becarful of that it might influence on the conversions**

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[267]:


countries = pd.read_csv('countries.csv')
countries.head()


# In[268]:


new = countries.set_index('user_id').join(df2.set_index('user_id'), how='inner')
new.head()


# In[269]:


# Check the unique countries
new.country.unique()


# In[270]:


new[['US', 'UK','CA']] = pd.get_dummies(new['country'])[['US', 'UK', 'CA']]
new.head()


# In[271]:


# Should include only two countries UK and US and the intercept (CA will be considered as the reference) 
log_mod = sm.Logit(new['converted'],new[['intercept','UK','US']])
summary_results = log_mod.fit()
summary_results.summary()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[272]:


new['UK_page'] = new['UK'] * new['ab_page']
new['US_page'] = new['US'] * new['ab_page']
new['CA_page'] = new['CA'] * new['ab_page']
new.head()


# In[273]:


log_mod_countries = sm.Logit(new['converted'],new[['intercept','ab_page', 'US', 'CA','US_page','CA_page']])
summary_results = log_mod_countries.fit()
summary_results.summary()


# ##### Based on the above results, we can notice that the P-value greater than %0.05 and less than 95%, It means there's no a significant effects on conversion between the page and countries.

# ### Based on findings in all the 3 main parts of the analysis using the data set(ab_data.csv), my advice for the concerned company it's there is no significance to implement the new page cause there's no considerable conversion rate between the old page and new page.

# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[274]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

