# --------------
import pandas as pd
import scipy.stats as stats
import math
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#Sample_Size
sample_size=2000

#Z_Critical Score
z_critical = stats.norm.ppf(q = 0.95)  


# path        [File location variable]

#Code starts here

#Reading file
data=pd.read_csv(path)

#Sampling the dataframe
data_sample = data.sample(n=sample_size, random_state=0)


#Finding the mean of the sample
sample_mean = data_sample['installment'].mean()

#Finding the standard deviation of the sample
sample_std = data_sample['installment'].std()

#Finding the margin of error
margin_of_error = z_critical * (sample_std/math.sqrt(sample_size))

#Finding the confidence interval
confidence_interval = (sample_mean - margin_of_error,
                       sample_mean + margin_of_error)  

print("Confidence interval:")
print(confidence_interval)

#Finding the true mean
true_mean=data['installment'].mean()

print(("True mean: {}".format(true_mean)))

#Code ends here


# --------------
import matplotlib.pyplot as plt
import numpy as np

#Different sample sizes to take
sample_size=np.array([20,50,100])

#Code starts here
#Creating different subplots
fig,axes=plt.subplots(3,1, figsize=(10,20))

#Running loop to iterate through rows
for i in range(len(sample_size)):
    
    #Initialising a list
    m=[]
    
    #Loop to implement the no. of samples
    for j in range(1000):
        
        #Finding mean of a random sample
        mean=data['installment'].sample(sample_size[i]).mean()
        
        #Appending the mean to the list
        m.append(mean)
        
        
    #Converting the list to series
    mean_series=pd.Series(m)   

    #Plotting the histogram for the series
    axes[i].hist(mean_series, normed=True)

    

#Displaying the plot
plt.show()


#Code ends here


# --------------
#Importing header files

from statsmodels.stats.weightstats import ztest

#Code starts here

# Removing the last character from the values in column
data['int.rate'] = data['int.rate'].map(lambda x: str(x)[:-1])

#Dividing the column values by 100
data['int.rate']=data['int.rate'].astype(float)/100



#Applying ztest for the hypothesis
z_statistic, p_value = ztest(x1=data[data['purpose']=='small_business']['int.rate'], value=data['int.rate'].mean(), alternative='larger')

print(('Z-statistic is :{}'.format(z_statistic)))
print(('P-value is :{}'.format(p_value)))

#Code ends here


# --------------
#Importing header files
from statsmodels.stats.weightstats import ztest

#Code starts here

#Applying ztest for the hypothesis
z_statistic, p_value = ztest(x1=data[data['paid.back.loan']=='No']['installment'], x2=data[data['paid.back.loan']=='Yes']['installment'])

print(('Z-statistic is :{}'.format(z_statistic)))
print(('P-value is :{}'.format(p_value)))

#Code ends here


# --------------
#Importing header files
from scipy.stats import chi2_contingency

#Critical value 
critical_value = stats.chi2.ppf(q = 0.95, # Find the critical value for 95% confidence*
                      df = 6)   # Df = number of variable categories(in purpose) - 1

#Code starts here
# Subsetting the dataframe
yes=data[data['paid.back.loan']=='Yes']['purpose'].value_counts()
no=data[data['paid.back.loan']=='No']['purpose'].value_counts()


#Concating yes and no into a single dataframe
observed=pd.concat([yes.transpose(),no.transpose()], 1,keys=['Yes','No'])

print(observed)

chi2, p, dof, ex = chi2_contingency(observed)


print("Critical value")
print(critical_value)


print("Chi Statistic")
print(chi2)


