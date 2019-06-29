# --------------
# Import packages
import numpy as np
import pandas as pd
from scipy.stats import mode 
 



# code starts here
bank = pd.read_csv(path)
categorical_var = bank.select_dtypes(include = 'object')
print(categorical_var)
numerical_var = bank.select_dtypes(include = 'number')
print(numerical_var)





# code ends here


# --------------
# code starts here
banks = bank.drop(['Loan_ID'], axis=1)
bank_mode = banks.mode().iloc[0]
banks.fillna(bank_mode,inplace = True)
#code ends here


# --------------
# Code starts here






avg_loan_amount = pd.pivot_table(banks, index=['Gender', 'Married', 'Self_Employed'], values='LoanAmount', aggfunc=np.mean)
print(avg_loan_amount)


# code ends here



# --------------
# code starts here




loan_approved_se = banks.loc[(banks['Self_Employed'] == 'Yes') & (banks['Loan_Status'] == 'Y'), ['Loan_Status']].count()
loan_approved_nse = banks.loc[(banks['Self_Employed'] == 'No') & (banks['Loan_Status'] == 'Y'), ['Loan_Status']].count()

percentage_se = ((loan_approved_se * 100) / 614)[0]
percentage_nse = ((loan_approved_nse * 100) / 614)[0]

print(percentage_nse)
print(percentage_se)

# code ends here


# --------------
# code starts here
loan_term = banks['Loan_Amount_Term'].apply(lambda x: x / 12 )
big_loan_term = len(loan_term[loan_term >= 25])




# code ends here


# --------------
# code starts here
loan_groupby = banks.groupby('Loan_Status')
print(loan_groupby)
loan_groupby = loan_groupby[['ApplicantIncome', 'Credit_History']]
print(loan_groupby)
mean_values = loan_groupby.mean()
print(mean_values)



# code ends here


