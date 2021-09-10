import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
df = pd.read_csv("datasets/persona.csv")
df.head()



#data summary
###########
#first 5 observations of persona dataset
df.head()
#information about data types and null values
df.info()
#number of rows,columns
df.shape
#variables of the persona dataset including the dependent one
df.columns
#is there any missing value in our dataframe?
df.isnull().values.any()
#dataframe overview with aggregation values in transpose
df.describe().T
#############

#unique sources and frequencies
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
#There are 2 unique sources "android" and "ios".
#2974 android, 2026 ios.
###############
#unique Prices
df["PRICE"].nunique()
df["PRICE"].unique()
#There are 6 unique prices.

############

##prices and Frequencies
df["PRICE"].value_counts()
"""
Price    Quantity of Sold App
29            1305
39            1260
49            1031
19            992
59            212
9             200

"""
#############
#number of sales by country
sales_by_country = df["COUNTRY"].value_counts()
sales_by_country



###########
#revenue of total sales by country
df.groupby("COUNTRY")[['PRICE']].aggregate("sum")

#################
#sales by source
df.groupby("SOURCE")["PRICE"].count()
"""
         PRICE
COUNTRY       
bra      51354
can       7730
deu      15485
fra      10177
tur      15689
usa      70225
"""


######################
#average value of sales by country
df.groupby("COUNTRY")["PRICE"].mean()
"""
COUNTRY
bra    34.327540
can    33.608696
deu    34.032967
fra    33.587459
tur    34.787140
usa    34.007264

"""


##############
#average value of sales by source
df.groupby("SOURCE")["PRICE"].mean()
"""
android    34.174849
ios        34.069102
"""
#############
#average value of prices by country and source
df.groupby(['COUNTRY','SOURCE'])[["PRICE"]].aggregate("mean").unstack()
"""
             PRICE           
SOURCE     android        ios
COUNTRY                      
bra      34.387029  34.222222
can      33.330709  33.951456
deu      33.869888  34.268817
fra      34.312500  32.776224
tur      36.229437  33.272727
usa      33.760357  34.371703
"""


#######################################################################
#what are the average earnings by country,source, sex and age?
###################################################
df.groupby(["COUNTRY","SOURCE","SEX","AGE"])["PRICE"].mean().head()

###################################
#in order to better observation for our output, we apply the sort_values sort_values method to the price by ascending.

######################################
agg_df = df.groupby([col for col in df.columns if col != 'PRICE']).mean()
agg_df.sort_values("PRICE", ascending = False, inplace = True)
agg_df.head()


#################################
#let's convert the names in the index to variable names
agg_df.reset_index(inplace = True)


####################
#let's create age ranges and assign them to a categorical value and add it to our dataframe.

bins = [0,18,23,30,40,70]
lab = ["0_18", "19_23", "24_30", "31_40", "41_70"]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"],bins,labels=lab)
agg_df.head()

##########################################
#now we need to create extensive new personas and unify those personas.
for row in agg_df.values:
    print(row)

[row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_" + row[2].upper() + "_" + row[5].upper() for row in agg_df.values]
agg_df.head()

#now we can get rid of the other variables, "customers_level_based" does all the classifications that we need and that the other variables do.
#let's assign agg_df only with "customers_level_based" and "PRICE".
agg_df = agg_df[["customers_level_based", "PRICE"]]
agg_df.head()

agg_df["customers_level_based"].value_counts()
#there is more than one from the same segment in our dataset, we should fix this. after groupby the segments, we should get the price averages and deduplicate the segments.

agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
#we need to remove customers_level_based from the index and turn it into a variable.
agg_df = agg_df.reset_index()
agg_df["customers_level_based"].value_counts()
#now the all the personas that we created are unique.
################################

#we want to create segments according to price. this segmentation will give us the opportunity to take action and strategize sales.
agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels = ["D","C","B", "A"])
agg_df.head()
#average values of segments by the price.
agg_df.groupby("SEGMENT")["PRICE"].mean()
#observation of segment C.
agg_df[agg_df["SEGMENT"] == "C"]

######################################################
#find the segment and average income of a 33-year-old android user Turkish woman.
new_user = "ANROID_FEMALE_TURKEY_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
agg_df.head()
#now let's look at the 35-year-old ios user French man.
new_user = "IOS_MALE_FRA_31_40"
agg_df[agg_df["customers_level_based"] == new_user]

"""
now we can predict how much our customers will earn on average. we can develop a marketing strategy using this segmentation.

"""