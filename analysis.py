import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import exp, sqrt

#################### Part 1: Data Analysis and Preprocessing ####################

df = pd.read_csv("training.csv")

# ignore all conflict diamonds
df = df[df['Known_Conflict_Diamond'] != True]
del df['Known_Conflict_Diamond']

# Q: is dimensions directly related to carats? lets find out!
# replace dimensions with volume (in mm^3)
def meas_to_volume(text):
    """
    take in string of format 'axbxc' and return the product of a,b,c
    """
    lengths = text.split('x')
    vol = 1
    for l in lengths:
        vol *= float(l)
    return vol
df['Measurements'] = df['Measurements'].apply(meas_to_volume)
# find correlation between the 2 variables
print(df.corr(numeric_only=True))
# correlation is .999 so its redundant
del df['Measurements']

# Delete sparse and not significant (low corr) data:
del df['Table']
del df['Cut']
del df['Depth']

# to convert scales to numbers, lets map them to values
# using .unique(), we find the set of values ['Good' 'Very good' ...]
# entries that miss either symmetry or polish are missing the other, so delete them
df = df[df['Symmetry'] != " "]
def rating_to_number(rating):
    mapping_dict = {"Excellent": 5,
                    "Execllent": 5,#for the typo
                    "Very good": 4,
                    "Good": 3,
                    "Fair": 2,
                    "Faint": 1,
                    " ": -1}
    return mapping_dict[rating]
df['Polish'] = df['Polish'].apply(rating_to_number)
df['Symmetry'] = df['Symmetry'].apply(rating_to_number)

# onto 'clarity' -- there is one row with missing value, and we want to delete it
df = df[pd.notna(df['Clarity'])]
# in the future, we can set the number to their average price or something(think bar chart) to represent jumps
def clarity_to_number(clarity):#maybe I should exclude the number part and just stick to letters
    mapping_dict = {'SI1': 5,
                    'SI2': 4,
                    'VS2': 6,
                    'VVS1': 8,
                    'VS1': 7,
                    'VVS2': 9,
                    'I1': 3,
                    'IF': 10,
                    'FL': 11,
                    'I2': 2,
                    'I3': 1}
    return mapping_dict[clarity]
df['Clarity'] = df['Clarity'].apply(clarity_to_number)

# onto color now. only 7 unique colors, so flag them with -1
# when purchasing diamonds, look ad ffancy darkbrown and flby
def color_to_number(color):
    omited_colors = {'Fdpink', 'Ffcg', 'Ffcy', 'Fiy', 'Fiyellow', 'Flyellow', 'Fvyellow', 'Ffancy darkbrown', 'Flby'}
    if color in omited_colors:
        return -1
    c = color[0].lower()#first letter chosen for ranges
    #lower case chars have ord = 97,98,...,121,122, so we convert d-z to 23-1
    return 123-ord(c)
df['Color'] = df['Color'].apply(color_to_number)
df = df[df['Color'] != -1]

# we convert categorical variables via one hot encoding
# cert only has nan, 'AGSL', and 'GemEx'
# watch out for multicollinearity
for cert in ['AGSL', 'GemEx']:
    df[cert] = (df['Cert'] == cert).astype(int)
del df['Cert']

# same thing for regions
# actually, regions doesnt change the model much, so lets omit it
# regions = ['Russia', 'South Africa', 'Botswana', 'Canada', 'DR Congo', 'Zimbabwe', 'Angola', 'Australia']
# for r in regions:
#     df[r] = (df['Regions'] == r).astype(int)
del df['Regions']

# and for shapes
shapes = {'Princess', 'Marquise', 'Round', 'Oval', 'Radiant', 'Emerald', 'Pear', 'Asscher', 'Cushion', 'Uncut'}
# fix the typos ('ROUND', 'Oval_', 'Marquis')
def correct_shapes(shape):
    if shape in shapes:
        return shape #correct shape
    if shape == "Oval ":
        return "Oval"
    elif shape == "Marquis":
        return "Marquise"
    elif shape == "ROUND":
        return "Round"
df['Shape'] = df['Shape'].apply(correct_shapes)
#after playing around the with model, I think just classifying diamonds as uncut or cut is good
df['Uncut'] = (df['Shape'] == 'Uncut').astype(int)
del df['Shape']

# finaly, we delete the id because it is loosely correlated with the data
del df['id']

#lets change the price variable to its square root and check results
df['LogPrice'] = df['LogPrice'].apply(lambda x: sqrt(exp(x)))
df['LogRetail'] = df['LogRetail'].apply(lambda x: sqrt(exp(x)))

df.to_csv('training_cleaned.csv')  

#################### Part 2: Fitting models ####################

# for the multiple regression model, we will make a model for each of the 4 vendors
df1 = df[df['Vendor'] == 1]
df2 = df[df['Vendor'] == 2]
df3 = df[df['Vendor'] == 3]
df4 = df[df['Vendor'] == 4]

def remove_cols(df):
    cols = ['Retail','LogRetail','Price','Vendor']
    for c in cols:
        del df[c]

for dfn in [df1, df2, df3, df4]:
	remove_cols(dfn)

#also have to remove confounding variables from the pricing model
for c in ['Retail','LogPrice','Price']:
    del df[c]

#we will use df again for analyzing the price it sold for
def make_model(df, var):
    #split into training and testing data
    dep_df = df[var]
    ind_df = df.loc[:, df.columns != var]
    x_train, x_test, y_train, y_test = train_test_split(ind_df,dep_df,test_size=0.2,train_size=0.8,random_state=22)
    #make a regression model
    model = LinearRegression().fit(x_train, y_train)
    # evaluate models on testing set
    # r_sq = model.score(x_test, y_test)
    # print(f"coefficient of determination: {r_sq}")# we are getting .85-.93!! 
    # print(f"intercept: {model.intercept_}")
    # print(f"coefficients: {model.coef_}")

    return model

# print("model 1:")
model1 = make_model(df1,'LogPrice')
# print("model 2:")
model2 = make_model(df2,'LogPrice')
# print("model 3:")
model3 = make_model(df3,'LogPrice')
# print("model 4:")
model4 = make_model(df4,'LogPrice')

# logretail might depend on vendor
price_model = make_model(df,'LogRetail')

#after running the models, we see that region and depth had little to no effect, so we are not going to consider it anymore

#################### Part 3: Cleaning the offers data ####################

### Now we can apply our models to predict the prices the diamonds can be bought and sold for
df_offers = pd.read_csv("offers.csv")

#proceed with the same steps as before, we have 
df_offers['Known_Conflict_Diamond'] = df_offers['Known_Conflict_Diamond'].apply(lambda x: int(x == True))
del df_offers['Measurements']
del df_offers['Table']
del df_offers['Cut']
del df_offers['Depth']
df_offers['Polish'] = df_offers['Polish'].apply(rating_to_number)
df_offers['Symmetry'] = df_offers['Symmetry'].apply(rating_to_number)
df_offers['Clarity'] = df_offers['Clarity'].apply(clarity_to_number)
df_offers['Color'] = df_offers['Color'].apply(color_to_number)
for c in ['AGSL', 'GemEx']:
    df_offers[c] = (df_offers['Cert'] == c).astype(int)
del df_offers['Cert']
del df_offers['Regions']
df_offers['Uncut'] = (df_offers['Shape'] == 'Uncut').astype(int)
del df_offers['Shape']  

#################### Part 4: Predicting prices/retail prices ####################

def pred_price(df):
    #here we check for the conditions:
    if df['Symmetry'] == -1 or df['Color'] == -1 or df['Known_Conflict_Diamond'] == 1:
        return 0
    vend_to_model = {1:model1,
                     2:model1,
                     3:model3,
                     4:model4}
    #we create different bids based on how good each of our models are (2-r^2)
    #i am a bit conservative on these bids so I will bid at 5% more
    buffer = {1: 1.13,
            2: 1.18,
            3: 1.17,
            4: 1.20}
    model = vend_to_model[df['Vendor']]
    # we need to fit the series accordingly
    data = df[['Carats','Clarity','Color','Polish','Symmetry','AGSL','GemEx','Uncut']]
    data = data.to_frame().transpose()
    #we can change the prices from log to normal since we are done with regression
    pred_sqrtprice = sqrt(buffer[df['Vendor']]) * model.predict(data)[0]
    return pred_sqrtprice**2
               
def pred_retail(df):
    #here we check for the conditions again:
    if df['Symmetry'] == -1 or df['Color'] == -1 or df['Known_Conflict_Diamond'] == 1:
        return 0
    #we only need these params
    data = df[['Carats','Clarity','Color','Polish','Symmetry','Vendor','AGSL','GemEx','Uncut']]
    data = data.to_frame().transpose()
    #we can change the prices from log to normal since we are done with regression
    pred_sqrtretail = price_model.predict(data)[0]
    return pred_sqrtretail**2

df_offers['pred_price'] = df_offers.apply(lambda x: pred_price(x), axis=1)
df_offers['pred_retail'] = df_offers.apply(lambda x: pred_retail(x), axis=1)

#plot pred_price against carats
# plt.close("all")
# df_offers.plot(x="Carats", y="pred_price", kind = "scatter")
# plt.show()

df_offers.to_csv('offers_pred.csv')

#################### Part 5: Let's go Diamond Shopping ####################

#now, iterate over the id, pred_price, and pred_retail and find the ones with the highest percent return
all_items = []
for _, row in df_offers.iterrows():
    if row['pred_price'] == 0:
        continue
    roi = (row['pred_retail']-row['pred_price'])/row['pred_price']
    if roi>0:
        all_items.append((row['id'],row['pred_price'],roi))
# we want to pick the most profitable diamonds first
all_items.sort(key = lambda x: x[2],reverse = True)
CAPITAL = 5000000
est_profit = 0
bids = []
# we iterate over all the diamonds, to exhaust our pockets because even a small diamond can be profitable
for diamond in all_items:
    if diamond[1] <= CAPITAL:
        bids.append(diamond)
        est_profit += diamond[2]*diamond[1]
        CAPITAL -= diamond[1]

# print(bids)
# print(CAPITAL)
# print(est_profit)#usually around 5 mil, not too shabby

#now we can fill in the offers csv with out actual offers (be careful with manipulating the orig data)
first_id = 8051
def get_row(id): #this is zero indexed
    return int(id - first_id)

final_df = pd.read_csv('offers.csv', index_col='id')
def set_bid(bid):
    id, price = bid[0], bid[1]
    final_df.iloc[get_row(id), final_df.columns.get_loc('Offers')] = price
for b in bids:
    set_bid(b)

final_df.to_csv('offers.csv')#be careful with overwriting the original data.
