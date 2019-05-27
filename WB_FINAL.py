#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:39:35 2018

@authors: 
Group 2 - Aashika Ashok, Lani Mateo, Marcelo Tournier, Qi Xue, Valentino Gaffuri Bedetta

Purpose: To run an exploratory data analysis of the World Bank Dataset 
(Region: Central Africa 1).


Presentation structure:
1. Title slide
2. Introduction
3. Methods (PESTE Analysis)
4. Political : 2 slides (Marcelo)
5. Economical : 2 slides (Valentino)
6. Social : 2 slides (Aashika)
7. Technological : 2 slides (Qi)
8. Environmental : 2 slides (Lani)
9. Conclusion

Naming conventions:

    The main dataframe will be named as "wb".
    
    Remember to use "wb.copy()" or "wb_<OTHERNAME>.copy()", to preserve the original dataframe
    Remember also to use underscore(_) instead of spaces, in their names.

"""

################################################################################## 

# Political analysis

################################################################################## 



# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%config IPCompleter.greedy=True


# import dataset
wb = pd.read_excel("world_data_hult_regions(1)_polit.xlsx")

# Getting info about the full dataset:
# identifying World Bank Regions:



print(wb.head())

print(wb.describe())

print(wb.info())

print(wb.Hult_Team_Regions.unique())

# correcting typo in 'Central Aftica 1':
wb = wb.replace('Central Aftica 1','Central Africa 1')

# subsetting df for 'Central Africa 1':

wb_group2 = wb.copy()[wb.Hult_Team_Regions == 
               'Central Africa 1'].sort_values('country_name')

print(wb.head())

print(wb.describe())

print(wb.info())

# New column - GDP in Billions of US Dollars:
wb_group2['gdp_bi_usd'] = round(wb_group2['gdp_usd']/1000000000,2)

# Identifying missing values:

print("Location of Missing Values:\n")

nas = pd.DataFrame(wb_group2.isnull().sum(),columns=["NaN_total"])
nans = pd.DataFrame(nas.NaN_total[nas.NaN_total>0].sort_values(ascending=False))
nans['NaN_percent'] = round(wb_group2.isnull().sum()/wb_group2.shape[0],2)

print(nans)
# NAs concentrated in Adult literacy rate, homicides, and tax revenue


# Flagging missing values:

wb_group2['m_adult_literacy_pct'] = wb_group2['adult_literacy_pct'].isnull().astype(int)
wb_group2['m_homicides_per_100k'] = wb_group2['homicides_per_100k'].isnull().astype(int)
wb_group2['m_tax_revenue_pct_gdp'] = wb_group2['tax_revenue_pct_gdp'].isnull().astype(int)
wb_group2['m_incidence_hiv'] = wb_group2['incidence_hiv'].isnull().astype(int)
wb_group2['m_compulsory_edu_yrs'] = wb_group2['compulsory_edu_yrs'].isnull().astype(int)

"""
Filling missing values for adult literacy rate:

Use linear regression model equation
y = -0.31x + 96.63
Where: 
y = 'adult_literacy_pct'
x= 'access_to_electricity_rural'

Find details in Footnote 1
Cross Validation score = 0.706455
mean std error = 10.162354
"""
for index in wb_group2.index:
    if pd.np.isnan(wb_group2['adult_literacy_pct'].loc[index]) == True:
        wb_group2['adult_literacy_pct'].loc[index] = -0.31*wb_group2['child_mortality_per_1k'].loc[index]+96.63

"""
MISSING VALUES FROM HOMICIDES:

Insert deaths per 100k, caused by Self-harm and interpersonal violence. 

Data from IHME (Institute for Health Metrics and Evaluation. U. Washington)
Ref - https://vizhub.healthdata.org/gbd-compare/

"""
wb_group2['homicides_selfharm_100k_IHME'] = [
7.690295288,
28.80888901,
13.64660827,
12.6082904,
14.84822956,
18.09191483,
7.929850777,
9.444180496,
17.27972687,
16.73539685,
9.466476865,
8.680058958,
57.58489952,
9.930242205]

# Creating copies of our dataframe with different imputation strategies:

# wb_group2_zero - NAs filled as '0':
wb_group2_zero  = pd.DataFrame.copy(wb_group2).fillna('0')

# wb_group2_median - NAs filled with Medians:
wb_group2_median = pd.DataFrame.copy(wb_group2)

for col in wb_group2_median.columns:
    if (wb_group2_median[col].isnull().any()) == True:
        wb_group2_median[col] = wb_group2_median[col].fillna(wb_group2_median[col].median())

# wb_group2_ext - NAs will be filled with external data
wb_group2_ext = pd.DataFrame.copy(wb_group2)



wb_group2[['country_name','income_group','homicides_per_100k','homicides_selfharm_100k_IHME']]


# Standardize data from electricity, homicides, edu, child mortality:
for col in ['access_to_electricity_pop',
            'internet_usage_pct',
            'compulsory_edu_yrs',
            'homicides_selfharm_100k_IHME',
            'child_mortality_per_1k',
           'adult_literacy_pct']:
    wb_group2_median[str(col+'_std')] = (wb_group2_median[col]-wb_group2_median[col].mean())/wb_group2_median[col].std()

wb_group2_median['gov_impact'] = wb_group2_median.access_to_electricity_pop_std+wb_group2_median.internet_usage_pct_std+wb_group2_median.compulsory_edu_yrs_std- wb_group2_median.homicides_selfharm_100k_IHME_std-wb_group2_median.child_mortality_per_1k_std

wb_group2_median[['compulsory_edu_yrs','homicides_selfharm_100k_IHME','child_mortality_per_1k']].describe()



###
### Government Slide 1
###

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.set(font_scale=1.5) 

sns.violinplot(data=wb_group2_median[['access_to_electricity_pop_std',
            'internet_usage_pct_std',
            'adult_literacy_pct_std',
            ]], 
               inner="points")    
plt.title("Government essentials: Infrastructure and Education (Standardized values)")
plt.text(1.1,2.4,'Cabo Verde')
plt.text(2.1,-2.4,"Cote d'Ivoire")

#sns.despine()
plt.axhline(y=0,color='red')
plt.subplot(1,2,2)
#plt.figure(figsize=(10,10))
sns.violinplot(data=wb_group2_median[[
            'homicides_selfharm_100k_IHME_std',
            'child_mortality_per_1k_std']], 
               inner="points",
              palette = 'cubehelix')    
plt.title("Government essentials: Health and Security (Standardized values)")
plt.text(0.1,3.2,'Sudan')
plt.text(1.1,-2.1,"Cabo Verde")
#sns.despine()
plt.axhline(y=0,color='red')
plt.tight_layout()
plt.savefig('gov1.png')
plt.show()

###
### Government Slide 2
###

# Import further data from the World Bank:
# Get from csv file, or from Footnote 2 in this script.

wb_political = pd.read_csv('political_polit.csv')

wb_political['income_group'] = wb_group2['income_group'].values
wb_political['gdp_bi_usd'] = wb_group2['gdp_bi_usd'].values
wb_political['women_in_parliament'] = wb_group2['women_in_parliament'].values
wb_political['internet_usage_pct'] = wb_group2['internet_usage_pct'].values


# Standardize data:
for col in wb_political.columns[1:5]:
    wb_political[str(col+'_std')] = (wb_political[col]-wb_political[col].mean())/wb_political[col].std()

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
sns.set(font_scale=1.5) 
plot2adata = wb_political[['stats_capacity_std','transparency_rating_std']]
plot2bdata = wb_political[['armed_forces_std','milit_exp_gdp_pct_std']]
sns.violinplot(data=plot2adata, inner="points")    
plt.title("Government essentials: Infrastructure and Education (Standardized values)")
plt.axhline(y=0,color='red')
plt.subplot(1,2,2)
sns.violinplot(data=plot2bdata, inner="points",palette = 'plasma')    
plt.title("Government essentials: Health and Security (Standardized values)")
plt.axhline(y=0,color='red')
plt.tight_layout()
plt.savefig('gov2.png')
plt.show()



plt.figure(figsize=(12,10))
sns.set(font_scale=1.2)
sns.despine()
sns.scatterplot(data=wb_political,
           x='stats_capacity',
                y='transparency_rating',
                hue='income_group',
                s=160,
               )
plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
sns.despine()
for line in range(0,wb_political.shape[0]):
     plt.text(wb_political.stats_capacity.iloc[line], 
             wb_political.transparency_rating.iloc[line]+pd.np.random.normal(0,0.05), 
             wb_political.country_name.iloc[line])
plt.title('Is Government transparency related with their official stats?')
plt.savefig('gov2a.png')
plt.show()

# Gov 2 stats capacity and military expenditure

col_x = 'stats_capacity'
col_y = 'milit_exp_gdp_pct'

sns.set(font_scale=1.2)
plt.figure(figsize=(18,20))
sns.despine()
sns.lmplot(data=wb_political,
           x=col_x,
                y=col_y,
           fit_reg = True,
                hue='income_group',
                #s=160,
          )
sns.despine()
#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#sns.despine()
#for line in range(0,wb_political.shape[0]):
#     plt.text(wb_political.country_name.iloc[line], 
             #wb_political[col_x].iloc[line]+pd.np.random.normal(0,0.05), 
             #wb_political[col_y].iloc[line])
#plt.title('Is Women in Parliament related with their official stats?')
#plt.xlabel (X.replace("_", " ")) # This will update based on variable defined  
#plt.ylabel (Y.replace("_", " "))
plt.savefig('gov2b.png')
plt.show()

col_x = 'stats_capacity'
col_y = 'milit_exp_gdp_pct'

sns.set(font_scale=1.0)
plt.figure(figsize=(10,10))
sns.despine()
sns.lmplot(data=wb_political,
           x=col_x,
                y=col_y,
           fit_reg = False,
                hue='income_group',
                #s=160,
          )
sns.despine()
#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#sns.despine()
for line in range(0,wb_political.shape[0]):
    plt.text(wb_political[col_x].iloc[line],wb_political[col_y].iloc[line]+pd.np.random.normal(-0.02,0.05),wb_political.country_name.iloc[line])
plt.title('Open to democracy, open to truth?')
plt.xlabel ('Stats Capacity') # This will update based on variable defined  
plt.ylabel ('Military Expenditure % GDP')
plt.savefig('gov2c.png')
plt.show()

# Relationship between GDP and stats_capacity:


col_x = 'transparency_rating'
col_y = 'internet_usage_pct'

sns.set(font_scale=1)
plt.figure(figsize=(10,10))
sns.despine()
sns.lmplot(data=wb_political,
           x=col_x,
                y=col_y,
           fit_reg = False,
                hue='income_group',
                #s=160,
          )
sns.despine()
for line in range(0,wb_political.shape[0]):
    plt.text(wb_political[col_x].iloc[line],wb_political[col_y].iloc[line]+pd.np.random.normal(-0.02,0.05),wb_political.country_name.iloc[line])

#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
#sns.despine()
#plt.title('Is Women in Parliament related with their official stats?')
plt.savefig('gov2d.png')
plt.show()


# Relationship with women in parliament and stats cap.
col_x = 'transparency_rating'
col_y = 'women_in_parliament'

plt.figure(figsize=(10,10))
sns.set(font_scale=1.2)
#sns.despine()
sns.lmplot(data=wb_political,
           x=col_x,
                y=col_y,
           #fit_reg = False,
                hue='income_group',
                #s=160,
               )
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.despine()
for line in range(0,wb_political.shape[0]):
    plt.text(wb_political[col_x].iloc[line],wb_political[col_y].iloc[line]+pd.np.random.normal(-0.02,0.05),wb_political.country_name.iloc[line])



plt.title('Is Stats Capacity related with # of women in parliament?')
plt.savefig('gov3.png')
plt.show()






# Pairplots:

politicalgrid = wb_political[['country_name',
                              'income_group',
                              'stats_capacity',
                              'transparency_rating',
                              'armed_forces',
                              'milit_exp_gdp_pct',
                             'women_in_parliament',
                              'gdp_bi_usd',
                             ]]
sns.pairplot(politicalgrid,
             hue='income_group',
             #size='gdp_bi_usd'
            )
plt.savefig('political_pairplots.png')
plt.show()

# Heatmap:
plt.figure(figsize=(10,10))
sns.set(font_scale=1.2)
sns.heatmap(politicalgrid.corr(),annot=True,cmap="GnBu")
plt.savefig('political_heatmap.png')
plt.show()

######################################################################
##
######################    E N D OF POLITICAL ANALYSIS    ############
##
######################################################################



################################################################################## 

# Economical analysis

################################################################################## 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dic 12 16:00hs 2018

@author: valentinogaffuribedetta
"


Purpose: To run an exploratory data analysis of the World Bank Dataset
(Region: Central Africa 1).


Presentation structure:
1. Title slide
2. Introduction
3. Methods (PESTE Analysis)
4. Political : 2 slides (Marcelo)
5. Economical : 2 slides (Valentino)
6. Social : 2 slides (Aashika)
7. Technological : 2 slides (Qi)
8. Environmental : 2 slides (Lani)
9. Conclusion

Naming conventions:

    The main dataframe will be named as "wb".
    
    Remember to use "wb.copy()" or "wb_<OTHERNAME>.copy()", to preserve the original dataframe
    Remember also to use underscore(_) instead of spaces, in their names.


"""


#### MARCELO'S CODE, STANDARIZED DATASET ###


# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%config IPCompleter.greedy=True
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# import dataset
wb = pd.read_excel("world_data_hult_regions_econ.xlsx")

# Getting info about the full dataset:
# identifying World Bank Regions:



print(wb.head())

print(wb.describe())

print(wb.info())

print(wb.Hult_Team_Regions.unique())

print(wb.shape)

# correcting typo in 'Central Aftica 1':
wb = wb.replace('Central Aftica 1','Central Africa 1')
wb.rename(columns={'CO2_emissions_per_capita)':'CO2_emissions_per_capita'}, 
                 inplace=True)

# subsetting df for 'Central Africa 1':

wb_group2 = wb[wb.Hult_Team_Regions ==
           	'Central Africa 1'].sort_values('country_name')

print(wb.head())

print(wb.describe())

print(wb.info())

print(wb.shape)

# Identifying missing values:

print("Location of Missing Values:\n")

nas = pd.DataFrame(wb_group2.isnull().sum(),columns=["NaN_total"])
nans = pd.DataFrame(nas.NaN_total[nas.NaN_total>0].sort_values(ascending=False))
nans['NaN_percent'] = round(wb_group2.isnull().sum()/wb_group2.shape[0],2)

print(nans)
# NAs concentrated in Adult literacy rate, homicides, and tax revenue


# Flagging missing values:

#wb_group2['m_adult_literacy_pct'] = wb_group2['adult_literacy_pct'].isnull().astype(int)
#wb_group2['m_homicides_per_100k'] = wb_group2['homicides_per_100k'].isnull().astype(int)
#wb_group2['m_tax_revenue_pct_gdp'] = wb_group2['tax_revenue_pct_gdp'].isnull().astype(int)
#wb_group2['m_incidence_hiv'] = wb_group2['incidence_hiv'].isnull().astype(int)
#wb_group2['m_compulsory_edu_yrs'] = wb_group2['compulsory_edu_yrs'].isnull().astype(int)

# loop for missing values #

for col in wb_group2:
    # creating columns with 0s for non missing values and 1s for missing values #
    if wb_group2[col].isnull().astype(int).sum()>0:
        wb_group2['m_'+col]=wb_group2[col].isnull().astype(int)
    else:
        print("""There is an error in the loop, check it !""")

print(wb_group2.info())

# Creating copies of our dataframe with different imputation strategies:

# wb_group2_zero - NAs filled as '0':
wb_group2_zero  = pd.DataFrame.copy(wb_group2).fillna('0')

# wb_group2_median - NAs filled with Medians:
wb_group2_median = pd.DataFrame.copy(wb_group2)

for col in wb_group2_median.columns:
    if (wb_group2_median[col].isnull().any()) == True:
        wb_group2_median[col] = wb_group2_median[col].fillna(wb_group2_median[col].median())

# wb_group2_ext - NAs will be filled with external data
wb_group2_ext = pd.DataFrame.copy(wb_group2)

"""
MISSING VALUES FROM HOMICIDES:

Insert deaths per 100k, caused by Self-harm and interpersonal violence.

Data from IHME (Institute for Health Metrics and Evaluation. U. Washington)
Ref - https://vizhub.healthdata.org/gbd-compare/

"""
wb_group2['homicides_selfharm_100k_IHME'] = [
7.690295288,
28.80888901,
13.64660827,
12.6082904,
14.84822956,
18.09191483,
7.929850777,
9.444180496,
17.27972687,
16.73539685,
9.466476865,
8.680058958,
57.58489952,
9.930242205]

wb_group2[['country_name','income_group','homicides_per_100k','homicides_selfharm_100k_IHME']]

"""
Filling missing values for adult literacy rate:

Use linear regression model equation
y = -0.31x + 96.63
Where: 
y = 'adult_literacy_pct'
x= 'access_to_electricity_rural'

Find details in Footnote 1
Cross Validation score = 0.706455
mean std error = 10.162354

"""
#wb_group2['adult_literacy_pct_ESTIM'] = -0.31*wb_group2['access_to_electricity_rural']+96.63   






#### VALENTINO'S CODE FOR THE ECONOMICAL ANALYSIS ###


# Complete dataset of the world bank #
wb_full=pd.read_excel('worldbank_sharks_full_econ.xlsx')
# with the general dataset #

wb_corr= wb.corr().round(4)
print(wb_corr)

fig, ax=plt.subplots(figsize=(20,20))
sns.set(font_scale=2)
sns.heatmap(wb_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_all_var')
plt.show()


# with Central Africa #

wb_group2_corr=wb_group2.corr().round(4)
print(wb_group2_corr)

fig, ax=plt.subplots(figsize=(20,20))
sns.set(font_scale=2)
sns.heatmap(wb_group2_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_CA')
plt.show()

# with Central Africa fill with median #
wb_group2_median_corr=wb_group2_median.corr().round(4)
print(wb_group2_median_corr)

fig, ax=plt.subplots(figsize=(20,20))
sns.set(font_scale=2)
sns.heatmap(wb_group2_median_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_CA_median')
plt.show()

#### Starting to take important variables in economical aspects ####

# Here we took the gdp_growth_pct to compare with some other variables using the filled missing values with the median #

# GDP growth corr with the CA df filled withe median #
wb_group2_median_corr.loc['gdp_growth_pct'].sort_values(ascending = False)
# Here we have some insights of negative corr

# GDP growth corr with the WB no filled #
wb_corr.loc['gdp_growth_pct'].sort_values(ascending = False)
# Weak correlation in the general dataset


# Here we took a look into a 2nd variable (exports_ptc_gdp) to see some insights #
# Exports growth corr with the CA df filled withe median #
wb_group2_median_corr.loc['exports_pct_gdp'].sort_values(ascending = False)
# here we have some insights of corr


# Exports growth corr with the WB no filled #
wb_corr.loc['exports_pct_gdp'].sort_values(ascending = False)
# Weak correlation in the general dataset


# Here we took a look into a 3er variable (fdi_ptc_gdp) to see some insights #
# FDI growth corr with the CA df filled withe median #
wb_group2_median_corr.loc['fdi_pct_gdp'].sort_values(ascending = False)
# weak corr, only 0.6280 with exports, seen on the exports analysis


# FDI growth corr with the WB no filled #
wb_corr.loc['fdi_pct_gdp'].sort_values(ascending = False)
# Weak correlation in the general dataset


# Here we took a look into a 4th variable (unemployment_pct) to see some insights #
# Unemployment growth corr with the CA df filled withe median #
wb_group2_median_corr.loc['unemployment_pct'].sort_values(ascending = False)
# weak corr, only 0.6280 with exports, seen on the exports analysis


# Unemployment growth corr with the WB no filled #
wb_corr.loc['unemployment_pct'].sort_values(ascending = False)
# Weak correlation in the general dataset


########################
# Scatterplots
########################
########################
# Adding subplots
########################


# Scatterplt of gdp_growth_pct
plt.subplot(2, 2, 1)

plt.scatter(x = 'access_to_electricity_rural',
            y = 'gdp_growth_pct',
            alpha = 0.7,
            color = 'red',
            data = wb_group2)


plt.title('Corr with GDP Growth')
plt.ylabel('gdp_growth_pct')
plt.xlabel('access_to_electricity_rural')
plt.grid(True)

########
# No much sense in the corr, seen in the graph

########################

# Scatterplt of gdp_growth_pct and exports_pct_gdp
plt.subplot(2, 2, 1)

plt.scatter(x = 'exports_pct_gdp',
            y = 'gdp_growth_pct',
            alpha = 0.7,
            color = 'red',
            data = wb_group2)


plt.title('Corr with GDP Growth')
plt.ylabel('gdp_growth_pct')
plt.xlabel('exports_pct_gdp')
plt.grid(True)

########
# Seems like closer conomies have more growht... rare event...

########################


wb_upgd=wb_group2.sort_values(ascending=False, by='gdp_growth_pct').head(3)
print(wb_upgd)

wb_upgd_corr=wb_upgd.corr()

#Heatmap
fig, ax=plt.subplots(figsize=(20,20))
sns.set(font_scale=2)
sns.heatmap(wb_upgd_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.5)

plt.savefig('correlation_matrix_high_gdp')
plt.show()

wb_upgd_corr.loc['gdp_growth_pct'].sort_values(ascending=False)
#
"""
Here we can see that this countries that have high rate of growht of GDP have the following var with high corr to explain why this growth:
tax_revenue_pct_gdp             1.000000 ++++++ when you fill nthing here
pct_female_employment           0.990432
pct_male_employment             0.990053
child_mortality_per_1k          0.946380
gdp_usd                         0.938195
exports_pct_gdp                 0.934592 ==> open country more growth
--- m_tax_revenue_pct_gdp           0.779646 ---
--- urban_population_pct            0.671253 ---

NOW COMPARE WITH COUNTRIES WITH SIMILAR GROWTH AND DO THE CORR TO SEE WHICH VAR WE HAVE AND THEN COMPARE.
FOCUS THE RESEARCH HERE.
"""

# Here I create a new DF with only the Central Africa Regions
# This is to compare the main to countries with countries in other region
wb[['country_name','Hult_Team_Regions','income_group','gdp_growth_pct']].sort_values(ascending=False, by='gdp_growth_pct').head(20)

wb_africa=wb[(wb['Hult_Team_Regions'] == 'Central Africa 1')|(wb['Hult_Team_Regions'] == 'Central Africa 2')].copy()

# Here we only see countries with the GDP growth column
wb_africa[['country_name','Hult_Team_Regions','income_group','gdp_growth_pct']].sort_values(ascending=False, by='gdp_growth_pct')

# Here we see thesum stats of GDP growth in Central Africa
wb_africa[['country_name','Hult_Team_Regions','income_group','gdp_growth_pct']].describe().sort_values(ascending=False, by='gdp_growth_pct')

###########
# Africa Region stats sum:
#
# gdp_growth_pct
# count	28.000000
# max	10.257493
# 75%	6.809933
# 50%	5.231712
# mean	4.908486
# 25%	2.860291
# std	2.740982
# min	0.415062

## here we will took countries with a GDP bigger than 7%

###########

# Here we only see onyly countries with GDP more than 7%countries with the GDP growth column
wb_africa_gdp=wb_africa[['country_name','Hult_Team_Regions','income_group','gdp_growth_pct']].sort_values(ascending=False, by='gdp_growth_pct').head(6)

### Here we can see that there are 6 countries with high GDP 7%+ and 3 for CA1 and 3 for CA2
# country_name	Hult_Team_Regions	income_group	gdp_growth_pct
#
# Ethiopia	Central Africa 2	Low income	10.257493
# Congo, Dem. Rep.	Central Africa 1	Lower middle income	9.470288
# Cote d'Ivoire	Central Africa 1	Lower middle income	8.794077
# Rwanda	Central Africa 1	Low income	7.624576
# Niger	Central Africa 2	Low income	7.529043
# Mali	Central Africa 2	Low income	7.043356
########

# Now let's compare the corr() on this countries
wb_africa_gdp.loc[:,'gdp_growth_pct'].sort_values(ascending = False)

# wb_median - NAs filled with Medians:
# Here we fill the missing values of CA1 and CA2 with the median because there is corr in the previous analysis but there were missing values:
wb_africa_median = pd.DataFrame.copy(wb_africa)

for col in wb_africa_median.columns:
    if (wb_africa_median[col].isnull().any()) == True:
        wb_africa_median[col] = wb_africa_median[col].fillna(wb_africa_median[col].median())
        

## Here we analyze the corr() again but with the missing values filled in
wb_africa_median_corr=wb_africa_median.corr().round(4)
wb_africa_median_corr['gdp_growth_pct'].sort_values(ascending=False)
wb_africa_median_corr['fdi_pct_gdp'].sort_values(ascending=False)
wb_group2_corr['fdi_pct_gdp'].sort_values(ascending=False)
wb_corr['fdi_pct_gdp'].sort_values(ascending=False)

## Analysing with gdp_growth_pct
wb_group2_median_corr['gdp_growth_pct'].sort_values(ascending=False)
wb_africa_median_corr['gdp_growth_pct'].sort_values(ascending=False)
wb_upgd_corr['gdp_growth_pct'].sort_values(ascending=False)


## Analysing with gdp_usd
wb_group2_median_corr['gdp_usd'].sort_values(ascending=False)
wb_africa_median_corr['gdp_usd'].sort_values(ascending=False)
wb_upgd_corr['gdp_usd'].sort_values(ascending=False)


########
"""
Takeaway afeter the analysis run it:

Economical assumption:
While there is a bidirectional relation between export and growth, there is no such effect from FDI to GDP; accordingly, positive effects of FDI on growth are realized by means of export. In other words, the impact of FDI on growth depends on the extent to which it boosts exports.

HERE WE HAVE 3 TOP COUNTRIES WITH GDP GROWTH AND 2 BEING THE LOWEST ON HTE GDP GROWTH, BUT THE MAIN COUNTRY IN THE ANALYSIS WILL BE CABO VERDE BECAUSE FOLLOWING THE ASSUMPTION THIS COUNTRY PRESENT A EXTRAGE BEHAVIOUR HAVING A HIGH FDI AND HIGH EXPORT BUT STILL HAVING A SMALL GROWTH IN THE GDP.

Cabo Verde:
This is not visible in Cabo Verde because of the following:
1) FDI= there is a change on the destination of the FDI, before 2012 the mayority where destinated to turism & Real State (77%), and then a small part went to other services like IT and Communications (11%), but from 2012 the destination of FDI changes, from 2012 turism & Real State had 36% and IT and Communications had 43% (soure: US. Dep. of State).

That previous info is aligned with the Internet Usage in PCT in Cabo verde of 40.26%, being the best country in the whole region of Central Africa 1.

To show this we will use a scatterplot comparing the access to internet, being Cabo Verde the best country on this, to show this change in the trend


Equatorial Guinea:
Dictatorship, no security, low stats, low transparency, they have a high rate of exports because of the oil, they are the second biggest country in oil production.


Congo, Dem. Rep.:

"""

###############
# Here we can find the charts explaining the assumption and takeaway explained before.


# Scatterplots comparing FDI, GDP and Exports

sns.set(font_scale=1.5)
plt.figure(figsize=(10,10)) 
col_x = 'exports_pct_gdp'
col_y = 'gdp_growth_pct'
sns.scatterplot(data= wb_group2_median,
                x=col_x,
                y=col_y,
                size='fdi_pct_gdp',
                sizes=(100,3000),
                legend=False
               )

for line in range(0,wb_group2_median.shape[0]):
    plt.text(wb_group2_median[col_x].iloc[line],wb_group2_median[col_y].iloc[line]+pd.np.random.normal(-0.05,0.05),wb_group2_median.country_name.iloc[line])

plt.text(5,0,'Size = FDI', fontsize=35)
plt.title ("""  GDP % - EXPORTS % of GDP
           FDI % of GDP""", fontsize=30)
plt.xlabel ('Exports % GDP')
plt.ylabel ('GDP Growth %')
plt.xlim(xmax=90)
plt.savefig('growth_exports_fdi.png')
plt.show()

#plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0, prop={'size': 20})


# Now we select the chart with the countries highest and loswest GDP growth:

wb_group2[['country_name','income_group','exports_pct_gdp','fdi_pct_gdp','gdp_growth_pct']].sort_values(ascending=False, by='gdp_growth_pct').copy().loc[(wb_group2['gdp_growth_pct'] > 7)|(wb_group2['gdp_growth_pct'] < 1)]

# Now here we do the scatter plot with the internet and transparency

wb_political = pd.read_csv('political_econ.csv')
wb_political['internet_usage_pct'] = wb_group2['internet_usage_pct'].values
wb_political['transparency_rating']

plt.figure(figsize=(10,10)) 
sns.scatterplot(x='transparency_rating',
               y='internet_usage_pct',
                size='internet_usage_pct',
                data=wb_political,
                sizes=(300,1000),
                legend=False
               )

plt.title ("""INTERNET USAGE / TRANSPARENCY""", fontsize=30)
plt.xlabel ('TRANSPARENCY')
plt.ylabel ('INTERNET USAGE')
#plt.xlim(xmax=90)
plt.savefig('INTERNET_TRANSPARENCY.png')
plt.show()

    
    
    
#sns.set(font_scale=1.5)
#plt.figure(figsize=(10,10))    
#sns.heatmap(wb_group2_median[['fdi_pct_gdp','exports_pct_gdp','gdp_growth_pct']],annot=True, cmap='plasma')




######################################################################
##
######################    E N D    ###################################
##
######################################################################



######################################################################
##
######################    E N D OF ECONOMICAL ANALYSIS    ############
##
######################################################################


################################################################################## 

# Social analysis

################################################################################## 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:39:35 2018

@authors: 
Group 2 - Aashika Ashok, Lani Mateo, Marcelo Tournier, Qi Xue, Valentino Gaffuri Bedetta

Purpose: To run an exploratory data analysis of the World Bank Dataset 
(Region: Central Africa 1).


Presentation structure:
1. Title slide
2. Introduction
3. Methods (PESTE Analysis)
4. Political : 2 slides (Marcelo)
5. Economical : 2 slides (Valentino)
6. Social : 2 slides (Aashika)
7. Technological : 2 slides (Qi)
8. Environmental : 2 slides (Lani)
9. Conclusion



"""
# import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%config IPCompleter.greedy=True


# import dataset
wb = pd.read_excel("world_data_hult_regions(1).xlsx")

# Getting info about the full dataset:
# identifying World Bank Regions:



print(wb.head())

print(wb.describe())

print(wb.info())

print(wb.Hult_Team_Regions.unique())

# correcting typo in 'Central Aftica 1':
wb = wb.replace('Central Aftica 1','Central Africa 1')

# subsetting df for 'Central Africa 1':

wb_group2 = wb[wb.Hult_Team_Regions == 
               'Central Africa 1'].sort_values('country_name')

print(wb.head())

print(wb.describe())

print(wb.info())

# New column - GDP in Billions of US Dollars:
wb_group2['gdp_bi_usd'] = round(wb_group2['gdp_usd']/1000000000,2)

# Identifying missing values:

print("Location of Missing Values:\n")

nas = pd.DataFrame(wb_group2.isnull().sum(),columns=["NaN_total"])
nans = pd.DataFrame(nas.NaN_total[nas.NaN_total>0].sort_values(ascending=False))
nans['NaN_percent'] = round(wb_group2.isnull().sum()/wb_group2.shape[0],2)

print(nans)
# NAs concentrated in Adult literacy rate, homicides, and tax revenue


# Flagging missing values:

wb_group2['m_adult_literacy_pct'] = wb_group2['adult_literacy_pct'].isnull().astype(int)
wb_group2['m_homicides_per_100k'] = wb_group2['homicides_per_100k'].isnull().astype(int)
wb_group2['m_tax_revenue_pct_gdp'] = wb_group2['tax_revenue_pct_gdp'].isnull().astype(int)
wb_group2['m_incidence_hiv'] = wb_group2['incidence_hiv'].isnull().astype(int)
wb_group2['m_compulsory_edu_yrs'] = wb_group2['compulsory_edu_yrs'].isnull().astype(int)

"""
MISSING VALUES FROM HOMICIDES:

Insert deaths per 100k, caused by Self-harm and interpersonal violence. 

Data from IHME (Institute for Health Metrics and Evaluation. U. Washington)
Ref - https://vizhub.healthdata.org/gbd-compare/

"""
wb_group2['homicides_selfharm_100k_IHME'] = [
7.690295288,
28.80888901,
13.64660827,
12.6082904,
14.84822956,
18.09191483,
7.929850777,
9.444180496,
17.27972687,
16.73539685,
9.466476865,
8.680058958,
57.58489952,
9.930242205]

# Creating copies of our dataframe with different imputation strategies:

# wb_group2_zero - NAs filled as '0':
wb_group2_zero  = pd.DataFrame.copy(wb_group2).fillna('0')

# wb_group2_median - NAs filled with Medians:
wb_group2_median = pd.DataFrame.copy(wb_group2)

for col in wb_group2_median.columns:
    if (wb_group2_median[col].isnull().any()) == True:
        wb_group2_median[col] = wb_group2_median[col].fillna(wb_group2_median[col].median())

# wb_group2_ext - NAs will be filled with external data
wb_group2_ext = pd.DataFrame.copy(wb_group2)



wb_group2[['country_name','income_group','homicides_per_100k','homicides_selfharm_100k_IHME']]


# Standardize data from electricity, homicides, edu, child mortality:
for col in ['access_to_electricity_pop',
            'internet_usage_pct',
            'compulsory_edu_yrs',
            'homicides_selfharm_100k_IHME',
            'child_mortality_per_1k']:
    wb_group2_median[str(col+'_std')] = (wb_group2_median[col]-wb_group2_median[col].mean())/wb_group2_median[col].std()

wb_group2_median['gov_impact'] = wb_group2_median.access_to_electricity_pop_std+wb_group2_median.internet_usage_pct_std+wb_group2_median.compulsory_edu_yrs_std- wb_group2_median.homicides_selfharm_100k_IHME_std-wb_group2_median.child_mortality_per_1k_std

wb_group2_median[['compulsory_edu_yrs','homicides_selfharm_100k_IHME','child_mortality_per_1k']].describe()


##############################################################################

# SOCIAL INDICATORS

# Statistical Capacity and Women in Parliament in Central Africa 1

wb_social = pd.read_csv('political.csv')

wb_social['income_group'] = wb_group2['income_group'].values
wb_social['women_in_parliament'] = wb_group2['women_in_parliament'].values


# lmplot of Statistical Capacity vs. Women in Parliment

col_x = 'stats_capacity'
col_y = 'women_in_parliament'

plt.figure(figsize=(15,15))
sns.set(font_scale=1.2)
sns.despine()
sns.lmplot(data=wb_social,
           x=col_x,
           y=col_y,
           fit_reg = True,
           hue='income_group',
           palette='plasma'
           )

plt.text(x=78, y=68, s='Rwanda')
sns.despine()

plt.title("""Is Statistical Capacity related to the 
Number of Women in Parliament?
          """, fontsize=18)

plt.xlabel('Statistical Capacity Score')
plt.ylabel('Number of Women in Parliment')

plt.savefig('social_1.png')
plt.show()


# Statistical Capacity and Women in Parliament in the world

wb_social_full = pd.read_excel('SOCIAL_COMPLETE.xlsx')

col_a = 'stats_capacity'
col_b = 'women_in_parliament'

plt.figure(figsize=(15,15))
sns.set(font_scale=1.2)
sns.despine()
sns.lmplot(data=wb_social_full,
           x=col_a,
           y=col_b,
           fit_reg = True,
           hue='income_group',
           palette='plasma'
           )

plt.text(x=79, y=66, s='Rwanda')
sns.despine()

plt.title("""Is Statistical Capacity related to the 
Number of Women in Parliament?
          """, fontsize=18)

plt.xlabel('Statistical Capacity Score')
plt.ylabel('Number of Women in Parliment')

plt.savefig('social_2.png')
plt.show()





######################################################################
##
######################    E N D OF SOCIAL ANALYSIS    ############
##
######################################################################

################################################################################## 

# Technological analysis

################################################################################## 


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:45:24 2018

@author: valentinogaffuribedetta
"""



### IMPORT LIBRARIES AND DATASET ###

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

file= 'world_data_hult_regions_tech.xlsx'
wb=pd.read_excel(file)



### CHECKING FOR MISSING VALUES IN GENERAL VIEW ###

print(
      wb.isnull()
      .any()
      )

# there are missing values #



### NOW LET'S COUNT THE AMOUNT OF MISSING VALUES IN GENERAL VIEW ###

print(
      wb[:]
      .isnull()
      .sum()      
      )



### CREATE ONLY FOR CENTRAL AFRICA 1 ###

ca= wb[wb['Hult_Team_Regions'] == 'Central Aftica 1']
print(ca)
ca



### FLAGGING MISSING VALUES ###

for col in ca:
    # creating columns with 0s for non missing values and 1s for missing values #
    if ca[col].isnull().astype(int).sum()>0:
        ca['m_'+col]=ca[col].isnull().astype(int)
    else:
        print("""There is an error in the loop, check it !""")

print(ca.head())

# check if the missing values are okay #
a=ca.isnull().sum().sum()
b=ca.iloc[:,-5:].sum().sum()

if a==b:
    print("It's Okay! The missing values match with the flagging!")
else:
    print("Oh, no... there is a problem with the falgging!")

    
    
### ANALYSIS OF THE SUBSET ###

print(ca.shape)
print(ca.info())
print(ca.describe())

print(ca[:].isnull().sum().sort_values(ascending=False))



### SEEING ONLY THE MISSING VALUES ###

# putting descending the values #
ca_missing = ca[:].isnull().sum().sort_values(ascending=False)
ca_missing = pd.DataFrame(ca_missing)

ca_missing= ca_missing[ca_missing[0] >0]
print(round(ca_missing,2))

# new coluimn for the pct #
ca_missing['missing_pct']=ca_missing/len(ca)
print(round(ca_missing,2))

# Create a filter to try to fill in the values with mean, median #
# And compare with others countries of the dataset, in this case we compare with the other countries that also have 'Low Income' cotegory #

## GENERAL WB ###
# lit= literacy #
wb_lit= wb[['income_group','adult_literacy_pct']]
wb_lit=wb_lit[wb_lit['income_group']=='Low income']
print(wb_lit)
print(wb_lit.isnull().sum())
print(wb_lit.notnull().sum())
print(wb_lit.shape)
print(round(wb_lit.describe(),2))
print(wb_lit.info())


# homicides per 100k # Here we have a min of 0 !!! and is not a missing value 
wb_hom=wb[['homicides_per_100k','income_group']]
wb_hom=wb_hom[wb_hom['income_group']=='Low income']
print(wb_hom)
print(wb_hom.isnull().sum())
print(wb_hom.notnull().sum())
print(wb_hom.shape)
print(round(wb_hom.describe(),2))
print(wb_hom.info())

# tax revenue pct od gdp #
wb_tax=wb[['tax_revenue_pct_gdp','income_group']]
wb_tax=wb_tax[wb_tax['income_group']=='Low income']
print(wb_tax.isnull().sum())
print(wb_tax.notnull().sum())
print(wb_tax.shape)
print(round(wb_tax.describe(),2))
print(wb_tax.info())

# incidence_hiv #
wb_hiv=wb[['incidence_hiv']]
print(wb_hiv.isnull().sum())
print(wb_hiv.notnull().sum())
print(wb_hiv.shape)
print(round(wb_hiv.describe(),2))
print(wb_hiv.info())

wb.boxplot(column=['incidence_hiv'])

# Compulsory_edu_yrs #
wb_edu=wb[['compulsory_edu_yrs']]
print(wb_edu.isnull().sum())
print(wb_edu.notnull().sum())
print(wb_edu.shape)
print(round(wb_edu.describe(),2))
print(wb_edu.info())

# COMPARE OUR REGION WITH A SIMILAR REGION #
# we are gonna use low income regions #

# Literacy #
print(wb_lit.groupby("income_group").mean())
print(wb_lit.groupby("income_group").median())
print(wb_lit.groupby('income_group').describe())


### COUNT THE AMOUNT OF COUNTRIES IN THE INCOME CATEGORIES ###
wb.groupby(['income_group'])['income_group'].count()

# Here we use two ways to try to have the mean and the median to fil in the values #
# print(wb_lit.groupby("Hult_Team_Regions").mean())
# print(wb_lit.groupby("Hult_Team_Regions").median())



# Here we can say that is better to use the income_group approach #
# The values of the mean and median are really similar so we use the mean to fill in #

# Here we count the amount of values that the mean and median calculations is taking in account, just to know if it is representative #



# violinplot for the incomegroup  #


wb_df= wb.copy()
plt.figure(figsize=(14,10))
sns.violinplot(x = 'income_group',
               y = 'access_to_electricity_pop',
               data = wb_df,
               orient = 'v')

plt.axhline(0, color = 'black')
plt.axhline(20 , color = 'red')
plt.axhline(100, color='purple')


plt.savefig('violin1')
plt.show()



#  the pie chart of the average of the pct_of the three industries(ca) #

pae = np.mean(ca.iloc[0:13,12]) 
pie = np.mean(ca.iloc[0:13,13]) 
pse = np.mean(ca.iloc[0:13,14]) 

labels   = ['pct_agriculture_employment', 
            'pct_industry_employment', 'pct_services_employment']

quants   = [pae , pie ,pse]


def draw_pie(labels,quants):
    
    plt.figure(1, figsize=(8,8))
    
    expl = [0,0.1,0]  
   
    colors  = ["green","blue","orange"] 
   
    plt.pie(quants, explode=expl, colors=colors, labels=labels, 
            autopct='%1.1f%%',pctdistance=0.4,labeldistance=0.6, shadow=False)
    plt.title('pct_of the three industries(ca)', bbox={'facecolor':'0.8', 'pad':5})
    plt.savefig('piechart1.png')  
    plt.show()

draw_pie(labels,quants)


#  the pie chart of the bias #


labels   = ['pct_agriculture_employment', 
            'pct_industry_employment', 'pct_services_employment']

quants   = [ca.iloc[10,12] , ca.iloc[10,13] ,ca.iloc[10,14]]


def draw_pie(labels,quants):
    
    plt.figure(1, figsize=(8,8))
    
    expl = [0,0.1,0]  
   
    colors  = ["green","blue","orange"] 
   
    plt.pie(quants, explode=expl, colors=colors, labels=labels, 
            autopct='%1.1f%%',pctdistance=0.4,labeldistance=0.6, shadow=False)
    plt.title('pct_of the three industries(Congo, Rep.)', bbox={'facecolor':'0.8', 'pad':5})
    plt.savefig('piechart2.png')   
    plt.show()

draw_pie(labels,quants)

# the world average of the pct_of the three industries # 


labels   = ['pct_agriculture_employment', 
            'pct_industry_employment', 'pct_services_employment']

quants   = [wb.iloc[217,12] , wb.iloc[217,13] ,wb.iloc[217,14]]


def draw_pie(labels,quants):
    
    plt.figure(1, figsize=(8,8))
    
    expl = [0,0.1,0]  
   
    colors  = ["green","blue","orange"] 
   
    plt.pie(quants, explode=expl, colors=colors, labels=labels, 
            autopct='%1.1f%%',pctdistance=0.5,labeldistance=0.6, shadow=False)
    plt.title('pct_of the three industries(world)', bbox={'facecolor':'0.8', 'pad':5})
    plt.savefig('piechart3.png')  
    plt.show()

draw_pie(labels,quants)

# lowincome group #   

wb_pae= wb[['income_group','pct_agriculture_employment']]
wb_pae=wb_pae[wb_pae['income_group']=='Low income']
print(wb_pae)
print(wb_pae.isnull().sum())
print(wb_pae.notnull().sum())
print(wb_pae.shape)
w_pae = np.mean(wb_pae.iloc[0:33, 1]) 


wb_pie= wb[['income_group','pct_industry_employment']]
wb_pie=wb_pie[wb_pie['income_group']=='Low income']
print(wb_pie)
print(wb_pie.isnull().sum())
print(wb_pie.notnull().sum())
print(wb_pie.shape)
w_pie = np.mean(wb_pie.iloc[0:33, 1]) 

wb_pse= wb[['income_group','pct_services_employment']]
wb_pse=wb_pse[wb_pse['income_group']=='Low income']
print(wb_pse)
print(wb_pse.isnull().sum())
print(wb_pse.notnull().sum())
print(wb_pse.shape)
w_pse = np.mean(wb_pse.iloc[0:33, 1]) 


quants   = [w_pae , w_pie ,w_pse]


def draw_pie(labels,quants):
    
    plt.figure(1, figsize=(8,8))
    
    expl = [0,0.1,0]  
   
    colors  = ["green","blue","orange"] 
   
    plt.pie(quants, explode=expl, colors=colors, labels=labels, 
            autopct='%1.1f%%',pctdistance=0.5,labeldistance=0.6, shadow=False)
    plt.title('pct_of the three industries(low income)', bbox={'facecolor':'0.8', 'pad':5})
    plt.savefig('piechart4.png')  
    plt.show()

draw_pie(labels,quants)


#   Lower middle income group #

wbm_pae= wb[['income_group','pct_agriculture_employment']]
wbm_pae=wbm_pae[wbm_pae['income_group']=='Lower middle income']
print(wbm_pae)
print(wbm_pae.isnull().sum())
print(wbm_pae.notnull().sum())
print(wbm_pae.shape)
wm_pae = np.mean(wbm_pae.iloc[0:46, 1]) 


wbm_pie= wb[['income_group','pct_industry_employment']]
wbm_pie=wbm_pie[wbm_pie['income_group']=='Lower middle income']
print(wbm_pie)
print(wbm_pie.isnull().sum())
print(wbm_pie.notnull().sum())
print(wbm_pie.shape)
wm_pie = np.mean(wbm_pie.iloc[0:46, 1]) 

wbm_pse= wb[['income_group','pct_services_employment']]
wbm_pse=wbm_pse[wbm_pse['income_group']=='Lower middle income']
print(wbm_pse)
print(wbm_pse.isnull().sum())
print(wbm_pse.notnull().sum())
print(wbm_pse.shape)
wm_pse = np.mean(wbm_pse.iloc[0:46, 1]) 


quants   = [wm_pae , wm_pie ,wm_pse]


def draw_pie(labels,quants):
    
    plt.figure(1, figsize=(8,8))
    
    expl = [0,0.1,0]  
   
    colors  = ["green","blue","orange"] 
   
    plt.pie(quants, explode=expl, colors=colors, labels=labels, 
            autopct='%1.1f%%',pctdistance=0.4,labeldistance=0.6, shadow=False)
    plt.title('pct_of the three industries(lower middle)', bbox={'facecolor':'0.8', 'pad':5})
    plt.savefig('piechart5.png')  
    plt.show()

draw_pie(labels,quants)





######################################################################
##
######################    E N D OF TECHNOLOGICAL ANALYSIS    ############
##
######################################################################

################################################################################## 

# Environmental analysis

################################################################################## 



# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:32:25 2018

@author: Lani L. Mateo

Working Directory:
C:/Users/DELL/OneDrive - Education First/Desktop/PyCourses

Purpose:
To explore World Bank 2014 data focusing on Central Africa 1 Region and identify
ways to:

1. Flag anomalies in the data set.
2. Present trends and outliers.
3. Deliver insights and recommendations to further improve data in the region.

"""
# Step 1: Import modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Import dataset
wb = pd.read_excel ('world_data_hult_regions(1)_env.xlsx')
wb_ext_env = pd.read_excel ('worldbank_sharks_full_env.xlsx')

# Step 3: Correct typo in data set
wb = wb.replace('Central Aftica 1','Central Africa 1')
wb.rename(columns={'CO2_emissions_per_capita)':'CO2_emissions_per_capita'}, 
                 inplace=True)

# Step 4: Subset wb for 'Central Africa 1':
wb_CA1 = wb[wb.Hult_Team_Regions ==
           	'Central Africa 1'].sort_values('country_name')

# Step 5: Identify missing values:
for col in wb:
    print(col)
    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """
    if wb[col].isnull().astype(int).sum() > 0:
        wb['m_'+col] = wb[col].isnull().astype(int)
        
# Step 6: Get info about the full dataset:
print(wb.head())
print(wb.describe())
print(wb.info())
print(wb.Hult_Team_Regions.unique())

# Step 7: Define unique Data Frames
wb_mean  = pd.DataFrame.copy(wb)
wb_median = pd.DataFrame.copy(wb_mean)
wb_dropped = pd.DataFrame.copy(wb_median)

# Step 8: Create Loops for mean, median or define dropped
# Loop for wb_mean
for col in wb_mean:
    """ Impute missing values using the mean of each column """
    if wb_mean.iloc[:,6].isnull().astype(int).sum() > 0:
        col_mean = wb_mean.iloc[:,6].mean()
        wb_mean.iloc[:,6:] = wb_mean.iloc[:,6:].fillna(col_mean).round(2)

# Loop for wb_median
for col in wb_median:
    """ Impute missing values using the median of each column """
    if wb_median.iloc[:,6].isnull().astype(int).sum() > 0:
        col_median = wb_median.iloc[:,6].median()
        wb_median.iloc[:,6:] = wb_median.iloc[:,6:].fillna(col_median).round(2)

# Dropna() for wb_dropped
wb_dropped = wb_dropped.dropna().round(2)

print(wb)

##############################################################################
# Step 9: Analyze Environment Indicators - Lani Mateo
# A. Corrections in dataframe
"""
Marce, please add the following code under the correction of typo errors:
    
wb.rename(columns = {'CO2_emissions_per_capita)':'CO2_emissions_per_capita'}, 
                 inplace = True)
"""
# B1. Exploring CO2 emissions per capita and Air Pollution
print (wb_CA1.CO2_emissions_per_capita.describe())
print (wb_CA1.avg_air_pollution.describe())

# B2. Which indicators correlate with the 2 in Central Africa 1 Region?
wb_CA1_corr = wb_CA1.corr().round(2).sort_values(
        by ='CO2_emissions_per_capita',ascending = False)

print (wb_CA1_corr)

plt.figure(figsize=(15,15))
sns.set(font_scale=1.5)
sns.heatmap(wb_CA1_corr,
            cmap = 'Blues',
            square = True,
            annot = False,
            linecolor = 'black',
            linewidths = 0.1)

plt.title ('Correlations (Central Africa 1 Region)')

"""
Based on the correlations, prioritize exploration of CO2 emissions with
>>> exports_pct_gdp (0.61).

HIV incidence and Homicides per 100k have 0.63 and 0.62 correlations
respectively but there isn't a story to tell unless otherwise, because these
are highly correlated to characteristics of urbanization. However, there is
weak correlation between CO2 emissions per capita and urban population and
urban growth percentage. It must be a spurious correlation but taking note of
it here just in case some story reveals itself in the future.
    
"""
# C1. Exploration of trends in CO2 emissions compared to other variables
'In the interest of time, I will focus exploratory analysis on CO2 emissions.'

# C2. Environment variable codes
"""
Studies show that key boosters of CO2 emissions are economic development and
energy efficiency. % of exports from GDP would be good indicator of economic
development. In terms of energy efficiency, maybe exploring access to electricty
in urban areas is a good start but need to explore other factors outside
of the data set like % share of agriculture industry in GDP since it is
the top 3 contributor in CO2 emissions.

Other variables than can be explored against CO2 emissions
'gdp_usd'
'urban_population_pct'
'urban_population_growth_pct'
'access_to_electricity_urban' 
'average_air_pollution'

'Exports of goods and services(% of GDP)'
'GDP(current US$)'
'Urban Population(% of total)'
'Urban population growth(annual%)'
'Access to electricity(% of population)'
'CO2 emissions per capita'

"""
# D. How to present analysis. 
# D1. Definition of X and Y axis, size of markers 
# Global variables for all plots D2a, D2b and D2c
X = 'exports_pct_gdp' # Replace to check other variables for X axis
Y = 'CO2_emissions_per_capita' # Replace to check other variables for Y axis
S = 'urban_population_growth_pct' # Replace to modify size of markers in scatterplot
T = 'Exports of goods and services(% of GDP)' # Replace to modify title
H = 'income_group'

# D2. Choose which bests represents your analysis. 
# D2a. Using lmplot
sns.set(font_scale=1.5)
A1 = sns.lmplot(x = X,
                y = Y,
                hue = "income_group",
                height=8,
                aspect= 1.5,
                fit_reg = True,
                scatter_kws={"s": 400},
                palette = 'plasma',
                data = wb_CA1)
A1._legend.set_title('Income Group')
plt.text (x = 30.18, y= 4.96, s = 'World')
#plt.plot([0, 4.96], [1.5, 0], linewidth=2) # If you want to draw a line on the chart
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
#plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.savefig('env_1.png')
plt.show()

# D2b. Using scatterplot
plt.figure(figsize=(14,8))
A2 = sns.scatterplot(x = X,
                     y = Y,
                     sizes = (80,500),                                    
                     hue = 'income_group',
                     size = S,
                     legend = 'brief',
                     palette = 'plasma',
                     data = wb_CA1)
A2.text(47.69, 4.7+(-.05),#change these values based on x,y coordinates of marker
            'Equatorial Guinea',
            horizontalalignment='left',
            size='medium',
            color='black',
            weight='medium')
plt.legend(loc=0, prop={'size': 15})
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.savefig('env_2.png')
plt.show()

# D2c. Using regplot
plt.figure(figsize=(14,8))
A3 = sns.regplot(x = X,
                 y = Y,
                 marker = 'o',
                 scatter_kws = {'s':500},
                 data = wb_CA1) 
A3.text(47.69, 4.7+(-.05),# change these values based on x,y coordinates of marker
           'Equatorial Guinea',# change based on annotation for specific marker
            horizontalalignment='left',
            size='medium',
            color='black',
            weight='light')    
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.savefig('env_3.png')
plt.show()


# D3. Using swarmplots
plt.figure(figsize=(14,8))
sns.stripplot(x = X,
              y = Y,
              data = wb,
              size = 10,
              orient = 'v')
plt.title ('CO2 emissions per capita and Exports % GDP') # Manually update
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable X defined  
plt.ylabel (Y.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('Exports of Goods and Services (% GDP)') # Use if indicators are abbreviated or mispelled
plt.savefig('env_4.png')
plt.show()


# D4. Using boxplot
plt.figure(figsize=(10,6))
plt.boxplot(x = Y,
            data = wb_CA1,
            vert = False,
            patch_artist = True,
            meanline = True,
            showmeans = True)
plt.xlabel (X.replace("_", " ").capitalize()) # This will automate based on variable Y defined
plt.xlabel ('CO2 emissions per capita') # Use if indicators are abbreviated or mispelled
plt.savefig('env_5.png')
plt.show()



######################################################################
##
######################    E N D OF ENVIRONMENTAL ANALYSIS    ############
##
######################################################################

#######################
# Footnotes:
#######################

###               ###
###               ###
###               ###
### Footnote 1 -  ###
#####################

# Linear regression model to estimate Adult Literacy Rate:

# For this purpose, we used a linear model with data from the whole World Bank Dataset, with all countries and all years available;

# To train the data, Download full World Bank dataset from 'http://databank.worldbank.org/data/download/WDI_csv.zip' (data in file 'WDIData.csv')

wb_full = pd.read_csv("WDIData.csv")

# Data is organized differently than our assignment dataset. We will need to transform it beforehand...

new_cols = ['EG.ELC.ACCS.ZS', 'EG.ELC.ACCS.RU.ZS', 'EG.ELC.ACCS.UR.ZS', 
            'EN.ATM.CO2E.PC', 'SE.COM.DURS', 'SL.FAM.WORK.FE.ZS', 
            'SL.FAM.WORK.MA.ZS', 'SL.AGR.EMPL.ZS', 'SL.IND.EMPL.ZS', 
            'SL.SRV.EMPL.ZS', 'NE.EXP.GNFS.ZS', 'BM.KLT.DINV.WD.GD.ZS', 
            'NY.GDP.MKTP.CD', 'NY.GDP.MKTP.KD.ZG', 'SH.HIV.INCD.ZS', 
            'IT.NET.USER.ZS', 'VC.IHR.PSRC.P5', 'SE.ADT.LITR.ZS', 
            'SH.DYN.MORT', 'EN.ATM.PM25.MC.M3', 'SG.GEN.PARL.ZS', 
            'GC.TAX.TOTL.GD.ZS', 'SL.UEM.TOTL.ZS', 'SP.URB.TOTL.IN.ZS', 
            'SP.URB.GROW']
wb_filter = wb_full[wb_full['Indicator Code'].isin(new_cols)]
wb_filter = wb_filter.melt(wb_filter[["Country Name","Country Code","Indicator Name","Indicator Code"]])
wb_filter = wb_filter.drop(axis=1,columns=["Country Code","Indicator Code"])
wb_filter = wb_filter.rename(columns={'variable':'Year','value':'Value'})

# Dropping unwanted "Unnamed: 62" year in value. It seems that of them has nans as values.
wb_filter = wb_filter[wb_filter['Year'] != 'Unnamed: 62']

# Creating the new dataframe:
dflist = list()
countrylist = wb_filter['Country Name'].unique()
yearlist = wb_filter['Year'].unique()
for country in countrylist:
    for year in yearlist:
        dflist.append([country,year])

wb_alldata = pd.DataFrame(dflist,columns=['Country name','Year']).sort_values(['Country name','Year'])
wb_alldata = wb_alldata.reset_index(drop=True)

col_list = wb_filter['Indicator Name'].unique()

for colname in col_list:
    wb_alldata[colname] = wb_filter[wb_filter['Indicator Name'] == colname].sort_values(['Country Name','Year']).Value.values

def linear_model(x_axis,y_axis):
    feature = x_axis
    label = y_axis
    Xy = wb_alldata[[feature,label]].dropna()
    X = Xy[[feature]]
    y = Xy[[label]]

    # Split dataset for training purposes
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

    #load model:
    from sklearn.linear_model import LinearRegression

    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)

    intercept = regression_model.intercept_[0]

    from sklearn.metrics import mean_squared_error
    y_predict = regression_model.predict(X_test)
    regression_model_mse = mean_squared_error(y_predict, y_test)
    
    # returns a list = ['feature','coefficient for X','intercept','Cross Valid. Score','mean squared error','mean std. error','fit line equation']
    return [feature,regression_model.coef_[0][0],intercept,regression_model.score(X_test, y_test),regression_model_mse,regression_model_mse**(1/2),
           str(round(regression_model.coef_[0][0],2))+'x'+' + '+str(round(intercept,2))]

def equation_list(variable):
    model_list = list([['feature','coefficient for X','intercept','Cross Valid. Score','mean squared error','mean std. error','fit line equation "y="']])
    for item in list(wb_alldata.columns.values[2:]):
        model_list.append(linear_model(item,variable))
    model_results = pd.DataFrame(model_list[1:],columns=model_list[0])
    model_results = model_results.sort_values('Cross Valid. Score',ascending=False)
    model_results = model_results.reset_index(drop=True)
    return model_results[1:]

# Get a linear equation list with the best fits:
equation_list('Literacy rate, adult total (% of people ages 15 and above)')

# This led us to choose "Child mortality rate" as a predictor of adult literacy.  This makes the model easier to explain, and with enough accuracy for imputation purposes.

###               ###
###               ###
###               ###
### Footnote 2 -  ###
#####################

# Get support data for political analysis:

wb_full_melt = wb_full.melt(wb_full[["Country Name","Country Code","Indicator Name","Indicator Code"]])
wb_full_melt = wb_full_melt.drop(axis=1,columns=["Country Code","Indicator Code"])
wb_full_melt = wb_full_melt.rename(columns={'variable':'Year','value':'Value'})
wb_full_melt = wb_full_melt[wb_full_melt['Year'] == '2014']
wb_full_melt = wb_full_melt[wb_full_melt['Year'] != 'Unnamed: 62']

dflist = list()
countrylist = wb_full_melt['Country Name'].unique()
yearlist = wb_full_melt['Year'].unique()
for country in countrylist:
    for year in yearlist:
        dflist.append([country,year])
        
wb_all2014 = pd.DataFrame(dflist,columns=['Country name','Year']).sort_values(['Country name','Year'])
wb_all2014 = wb_all2014.reset_index(drop=True)
col_list = wb_full_melt['Indicator Name'].unique()

# Rebuilding the data frame:
n=1
for colname in col_list:
    print('column',n,'of 1,600 -',colname)
    n+=1
    wb_all2014[colname] = wb_full_melt[wb_full_melt['Indicator Name'] == colname].sort_values(['Country Name','Year']).Value.values

wb_g2_2014 = wb_all2014[wb_all2014['Country name'].isin(wb_group2.country_name.unique())]

# Getting columns for analysis:
col1 = 'Statistical Capacity score (Overall average)'
col2 = 'CPIA transparency, accountability, and corruption in the public sector rating (1=low to 6=high)'
col3 = 'Armed forces personnel, total'
col4 = 'Military expenditure (% of GDP)'


wb_political = wb_g2_2014.copy()[['Country name',col1,col2,col3,col4]]

# Rename columns:
wb_political = wb_political.rename({'Country name':'country_name',
                     'Statistical Capacity score (Overall average)':'stats_capacity',
                     'CPIA transparency, accountability, and corruption in the public sector rating (1=low to 6=high)':'transparency_rating',
                     'Armed forces personnel, total':'armed_forces',
                    'Military expenditure (% of GDP)':'milit_exp_gdp_pct'}, axis='columns')

#Flag NAs:
for col in wb_political.columns[1:]:
    wb_political[col+'_m'] = wb_political[col].isnull().astype(int)
    
# Filling NAs with medians:
for col in wb_political.columns[1:]:
    wb_political.loc[:,col] = wb_political[col].fillna(wb_political[col].median())

# saving as "political.csv":
wb_political.to_csv('political.csv',index=None)
