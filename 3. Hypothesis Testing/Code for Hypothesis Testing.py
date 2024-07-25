# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:38:41 2023

@author: kaush
"""

import pandas as pd
import scipy.stats as stats
#two sample 2 tail test
Cutlets=pd.read_csv("Cutlets.csv")
Cutlets
stats,pval=stats.ttest_ind(Cutlets['Unit A'],Cutlets['Unit B'])
print("Z calculated value: ",stats)
print("P Value: ",pval)

#ANOVA test
LABTAT = pd.read_csv('LabTAT.csv')
LABTAT
fvalue, pvalue = stats.f_oneway(LABTAT['Laboratory 1'], LABTAT['Laboratory 2'], LABTAT['Laboratory 3'], LABTAT['Laboratory 4'])
print("F calculated value: ",fvalue)
print("P Value: ",pvalue)

#chi square test
BR=pd.read_csv('BuyerRatio.csv')
table=BR.iloc[:,1:6]
table
stats.chi2_contingency(table)

#chi square test

from scipy.stats import chi2_contingency
from scipy.stats import chi2
custom= pd.read_csv('Costomer+OrderForm.csv')
custom.head()
print(custom['Phillippines'].value_counts(),
      custom['Indonesia'].value_counts(),
      custom['Malta'].value_counts(),
      custom['India'].value_counts())
observed=([[271,267,269,280],[29,33,31,20]])
observed
stat, p, dof, expected = chi2_contingency([[271,267,269,280],[29,33,31,20]])
stat
p
print('dof=%d' % dof)
print(expected)
alpha = 0.05
prob=1-alpha
critical = chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
	print('Dependent (reject H0),variables are related')
else:
	print('Independent (fail to reject H0), variables are not related')
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
	print('Dependent (reject H0)')
else:
	print('Independent (fail to reject H0)')