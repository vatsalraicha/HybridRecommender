# %% [markdown]
# # Hybrid Recommendation using LightFM

# %% [markdown]
# # Implementation

# %% [markdown]
# # Importing the required libraries

# %%
import pandas as pd # pandas for data manipulation
import numpy as np # numpy for sure
from scipy.sparse import coo_matrix # for constructing sparse matrix
# lightfm
from lightfm import LightFM # model
from lightfm.evaluation import auc_score

# timing
import time

# %% [markdown]
# # Importing the data from excel

# %%
# import the data
order=pd.read_excel('../data/Rec_sys_data.xlsx','order')
customer=pd.read_excel('../data/Rec_sys_data.xlsx','customer')
product=pd.read_excel('../data/Rec_sys_data.xlsx','product')

# %% [markdown]
# # Merging the datasets

# %%
# merge the data
full_table=pd.merge(order,customer,left_on=['CustomerID'], right_on=['CustomerID'], how='left')
full_table=pd.merge(full_table,product,left_on=['StockCode'], right_on=['StockCode'], how='left')

# %%
# Binning the ages into the proposed categories
bins = [18, 24, 34, 44, 54, np.inf]
labels = ['18-24', '25-34', '35-44', '45-54', '55+']
customer['Age Group'] = pd.cut(customer['Age'], bins=bins, labels=labels, right=False)

# Merge the customer age group information back to the full table
full_table = pd.merge(full_table, customer[['CustomerID', 'Age Group']], on='CustomerID', how='left')

# %%
"Order Shape", order.shape, "Customer Shape", customer.shape, "Product Shape", product.shape, "Full Table Shape", full_table.shape

# %%
# check for first 5 rows for order data
order.head()

# %%
# check for first 5 rows for customer data
customer.head()

# %%
customer["Customer Segment"].unique(), customer["Income"].unique(), customer["Gender"].unique()

# %%
from matplotlib import pyplot as plt
import seaborn as sns

# %%
from cycler import cycler

# Setting the color cycle for plotting
plt.rcParams['axes.prop_cycle'] = cycler(color=['#10A37F', '#147960', '#024736'])

# Setting the backend for inline display, suitable for Jupyter notebooks or similar environments
plt.rcParams['backend'] = 'module://matplotlib_inline.backend_inline'

# Enabling grid by default on axes
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.grid.axis'] = 'both'
plt.rcParams['axes.grid.which'] = 'major'

# Setting axes appearance
plt.rcParams['axes.edgecolor'] = (0.1, 0.1, 0.1)
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.labelcolor'] = (0.1, 0.1, 0.1)
plt.rcParams['axes.labelsize'] = 12.0
plt.rcParams['axes.titlecolor'] = 'auto'
plt.rcParams['axes.titlesize'] = 16.0

# Configuring the appearance of ticks
plt.rcParams['xtick.color'] = (0.1, 0.1, 0.1)
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.labelsize'] = 12.0
plt.rcParams['ytick.color'] = (0.1, 0.1, 0.1)
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.labelsize'] = 12.0

# %%
# Setting up the matplotlib figure
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
fig.tight_layout(pad=5.0)

# Plotting histograms for Order Data
order['Quantity'].plot(kind='hist', bins=30, ax=axes[0, 0], title='Order Quantity Distribution')
order['Discount%'].plot(kind='hist', bins=30, ax=axes[0, 1], title='Discount% Distribution')

# Plotting histograms for Customer Data
customer['Age'].plot(kind='hist', bins=30, ax=axes[1, 0], title='Customer Age Distribution')

# Plotting histograms for Product Data
product['Unit Price'].plot(kind='hist', bins=30, ax=axes[2, 0], title='Product Unit Price Distribution')

# Hiding empty subplots
axes[1, 1].axis('off')
axes[2, 1].axis('off')

plt.show()


# %% [markdown]
# * **Order Quantity Distribution**: The majority of orders contain a relatively small quantity of items, as evidenced by the sharp peak at the lower end of the scale. There are very few large quantity orders, which indicates that bulk purchases are rare.
# 
# * **Discount% Distribution**: The distribution of discounts is fairly uniform across the range, with a slight increase in frequency as the discount percentage increases. This could suggest that larger discounts are more commonly offered or that orders with larger discounts are more frequent.
# 
# * **Customer Age Distribution**: The age of customers appears to be fairly evenly distributed, with slight increases around the ages of 30 and 50. There are no particularly dominant age groups, which suggests a diverse customer base in terms of age.
# 
# * **Product Unit Price Distribution**: There is a large spike in the number of products with a very low unit price, indicating that most products are priced on the lower end. The frequency drops significantly for higher-priced items, showing that expensive products are much less common.

# %%
# check for first 5 rows for product data
product.head()

# %%
# Creating the list of unique users
def unique_users(data, column):
    return np.sort(data[column].unique())

# Creating the list of unique produts
def unique_items(data, column):
    item_list = data[column].unique()
    return item_list

def features_to_add(customer, column1,column2,column3):
    customer1 = customer[column1]
    customer2 = customer[column2]
    customer3 = customer[column3]
    return pd.concat([customer1,customer3,customer2], ignore_index = True).unique()

# Create id mappings to convert user_id, item_id, and feature_id
def mapping(users, items, features):
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(users):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id

    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(items):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id

    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(features):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id


    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping



# %%
# create the user, item, feature lists
users = unique_users(order, "CustomerID")
items = unique_items(product, "Product Name")
features = features_to_add(customer,'Customer Segment',"Age Group","Gender")

# %%
pd.DataFrame(users).head()

# %%
pd.DataFrame(items).head()

# %%
pd.DataFrame(features).head(20)

# %%
# generate mapping, LightFM library can't read other than (integer) index
user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping, \
           feature_to_index_mapping, index_to_feature_mapping = mapping(users, items, features)

# %%
full_table.head()

# %%
user_to_product_rating_train=full_table[['CustomerID','Product Name','Quantity']]

# %%
product_to_feature=full_table[['Product Name','Customer Segment','Quantity', "Gender","Age Group"]]

# %%
user_to_product_rating_train=user_to_product_rating_train.groupby(['CustomerID','Product Name']).agg({'Quantity':'sum'}).reset_index()

# %%
user_to_product_rating_train.tail()

# %% [markdown]
# # train test split

# %%
from sklearn.model_selection import train_test_split

# %%
# perform train test split - 67:33 percent
user_to_product_rating_train,user_to_product_rating_test = train_test_split(user_to_product_rating_train,test_size=0.33, random_state=42)

# %%
# check the shape of train data
user_to_product_rating_train.shape

# %%
# check the shape of test data
user_to_product_rating_test.shape

# %%
product_to_feature = product_to_feature.groupby(['Product Name', 'Customer Segment', 'Gender', 'Age Group']).agg({'Quantity': 'sum'}).reset_index()

# %%
# perform groupby
#product_to_feature=product_to_feature.groupby(['Product Name','Customer Segment']).agg({'Quantity':'sum'}).reset_index()

# %%
product_to_feature.head()

# %%
# create a function for interactions
def interactions(data, row, col, value, row_map, col_map):

    row = data[row].apply(lambda x: row_map[x]).values
    col = data[col].apply(lambda x: col_map[x]).values
    value = data[value].values

    return coo_matrix((value, (row, col)), shape = (len(row_map), len(col_map)))


# %%
# Update feature_to_index_mapping to include 'Age Group' and 'Gender'
all_features = np.concatenate([
    product_to_feature['Customer Segment'].unique(),
    product_to_feature['Age Group'].unique(),
    product_to_feature['Gender'].unique()
])
# Remove any potential nan values and ensure uniqueness
all_features = np.unique(all_features[~pd.isnull(all_features)])

# Generate a new complete feature mapping
complete_feature_to_index_mapping = {feature: index for index, feature in enumerate(all_features)}

# %%
# generate user_item_interaction_matrix for train data
user_to_product_interaction_train = interactions(user_to_product_rating_train, "CustomerID",
                                                    "Product Name", "Quantity", user_to_index_mapping, item_to_index_mapping)

# %%
# Generate item_to_feature interaction with Customer Segment using updated mappings
product_to_feature_interaction_segment = interactions(
    product_to_feature,
    "Product Name",
    "Customer Segment",
    "Quantity",
    item_to_index_mapping,
    complete_feature_to_index_mapping
)

# Generate item_to_feature interaction with Age Group using updated mappings
product_to_feature_interaction_age_group = interactions(
    product_to_feature,
    "Product Name",
    "Age Group",
    "Quantity",
    item_to_index_mapping,
    complete_feature_to_index_mapping
)

# Generate item_to_feature interaction with Gender using updated mappings
product_to_feature_interaction_gender = interactions(
    product_to_feature,
    "Product Name",
    "Gender",
    "Quantity",
    item_to_index_mapping,
    complete_feature_to_index_mapping
)

# %%
# generate user_item_interaction_matrix for test data
user_to_product_interaction_test = interactions(user_to_product_rating_test, "CustomerID",
                                                    "Product Name", "Quantity", user_to_index_mapping, item_to_index_mapping)

# %%
user_to_product_interaction_train

# %%
user_to_product_interaction_test

# %%
product_to_feature_interaction_segment, product_to_feature_interaction_age_group, product_to_feature_interaction_gender

# %%
from scipy.sparse import hstack

# Combining the item feature interaction matrices by stacking them horizontally
combined_item_features = hstack([product_to_feature_interaction_segment,
                                 product_to_feature_interaction_age_group,
                                 product_to_feature_interaction_gender])

# Checking the shape of the combined item features matrix
combined_item_features.shape

# %% [markdown]
# # Model building on training set
# Parameters:
# Loss=warp
# epochs=1
# num_threads=4

# %%
# initialising model with warp loss function
model_with_features = LightFM(loss = "warp")

# fitting the model with hybrid collaborative filtering + content based (product + features)
start = time.time()
#===================


model_with_features.fit_partial(user_to_product_interaction_train,
          user_features=None,
          item_features=combined_item_features,
          sample_weight=None,
          epochs=100,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# %%
start = time.time()
#===================
auc_with_features = auc_score(model = model_with_features,
                        test_interactions = user_to_product_interaction_test,
                        train_interactions = user_to_product_interaction_train,
                        item_features = combined_item_features,
                        num_threads = 4, check_intersections=False)
#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))



# %% [markdown]
# # Model building on training set
# Parameters:
# Loss=logistic
# epochs=1
# num_threads=4

# %%
# initialising model with warp loss function
model_with_features = LightFM(loss = "logistic")

# fitting the model with hybrid collaborative filtering + content based (product + features)
start = time.time()
#===================


model_with_features.fit_partial(user_to_product_interaction_train,
          user_features=None,
          item_features=combined_item_features,
          sample_weight=None,
          epochs=100,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# %%
start = time.time()
#===================
auc_with_features = auc_score(model = model_with_features,
                        test_interactions = user_to_product_interaction_test,
                        train_interactions = user_to_product_interaction_train,
                        item_features = combined_item_features,
                        num_threads = 4, check_intersections=False)
#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))


# %% [markdown]
# # Model building on training set
# Parameters:
# Loss=bpr
# epochs=1
# num_threads=4

# %%
# initialising model with warp loss function
model_with_features = LightFM(loss = "bpr")

# fitting the model with hybrid collaborative filtering + content based (product + features)
start = time.time()
#===================


model_with_features.fit_partial(user_to_product_interaction_train,
          user_features=None,
          item_features=combined_item_features,
          sample_weight=None,
          epochs=100,
          num_threads=4,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# %%
start = time.time()
#===================
auc_with_features = auc_score(model = model_with_features,
                        test_interactions = user_to_product_interaction_test,
                        train_interactions = user_to_product_interaction_train,
                        item_features = combined_item_features,
                        num_threads = 4, check_intersections=False)
#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))


# %% [markdown]
# # Model building on training set
# Parameters:
# Loss=logistic
# epochs=10
# num_threads=20

# %%
model_with_features = LightFM(loss = "logistic")

# fitting the model with hybrid collaborative filtering + content based (product + features)
start = time.time()
#===================


model_with_features.fit_partial(user_to_product_interaction_train,
          user_features=None,
          item_features=combined_item_features,
          sample_weight=None,
          epochs=100,
          num_threads=20,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# %%
start = time.time()
#===================
auc_with_features = auc_score(model = model_with_features,
                        test_interactions = user_to_product_interaction_test,
                        train_interactions = user_to_product_interaction_train,
                        item_features = combined_item_features,
                        num_threads = 4, check_intersections=False)
#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

print("average AUC without adding item-feature interaction = {0:.{1}f}".format(auc_with_features.mean(), 2))


# %%
def train_test_merge(training_data, testing_data):

    # initialising train dict
    train_dict = {}
    for row, col, data in zip(training_data.row, training_data.col, training_data.data):
        train_dict[(row, col)] = data

    # replacing with the test set

    for row, col, data in zip(testing_data.row, testing_data.col, testing_data.data):
        train_dict[(row, col)] = max(data, train_dict.get((row, col), 0))


    # converting to the row
    row_list = []
    col_list = []
    data_list = []
    for row, col in train_dict:
        row_list.append(row)
        col_list.append(col)
        data_list.append(train_dict[(row, col)])

    # converting to np array

    row_list = np.array(row_list)
    col_list = np.array(col_list)
    data_list = np.array(data_list)

    return coo_matrix((data_list, (row_list, col_list)), shape = (training_data.shape[0], training_data.shape[1]))

# %%
user_to_product_interaction = train_test_merge(user_to_product_interaction_train,
                                                 user_to_product_interaction_test)

# %%
user_to_product_interaction

# %%


# %%
from scipy.sparse import coo_matrix, csr_matrix

# Assuming `customer_df` is your DataFrame from the provided sample
# One-hot encode categorical features (including 'Gender', 'Customer Segment', 'Age Group', 'Income')
# Pandas get_dummies is a convenient way to do this directly from a DataFrame
user_features_df = pd.get_dummies(customer, columns=['Gender', 'Income', 'Customer Segment', 'Age Group'], dtype=int)


# %%
user_features_df

# %%
# Now convert this DataFrame into a sparse COO matrix
user_features_coo = coo_matrix(user_features_df.reset_index(drop=True))

# Convert COO matrix to CSR format for LightFM
user_features_csr = user_features_coo.tocsr()


# %% [markdown]
# # Final Model after combining the train and test data
# Parameters:
# Loss=warp
# epochs=10
# num_threads=20

# %%
# retraining the final model with combined dataset

final_model = LightFM(loss = "logistic",no_components=30)

# fitting to combined dataset

start = time.time()
#===================

final_model.fit(user_to_product_interaction,
          user_features=user_features_csr,
          item_features=combined_item_features,
          sample_weight=None,
          epochs=500,
          num_threads=10,
          verbose=False)

#===================
end = time.time()
print("time taken = {0:.{1}f} seconds".format(end - start, 2))

# %%
def get_recommendations(model,user,items,user_to_product_interaction_matrix,user2index_map,product_to_feature_interaction_matrix):

    # getting the userindex

        userindex = user2index_map.get(user, None)

        if userindex == None:
            return None

        users = userindex

        # products already bought

        known_positives = items[user_to_product_interaction_matrix.tocsr()[userindex].indices]
        print('User index =',users)

        # scores from model prediction
        scores = model.predict(user_ids = users, item_ids = np.arange(user_to_product_interaction_matrix.shape[1]),item_features=product_to_feature_interaction_matrix)

        # top items

        #top_items = items[np.argsort(-scores)]

        # Normalize the scores to be between 0 and 1
        min_score = np.min(scores)
        max_score = np.max(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score) if max_score != min_score else np.zeros_like(scores)
        # Sort the scores and get the indices of the sorted scores
        sorted_indices = np.argsort(-normalized_scores)

        # printing out the result
        print("User %s" % user)
        print("     Known positives:")  # already known products

        for x in known_positives[:10]:
            print("                  %s" % x)


        print("     Recommended:")  # products that are reccommended to the user

        #for x in top_items[:10]:
        #    print("                  %s" % x)
        for index in sorted_indices[:10]:  # Iterate over the indices of the top items
            item_name = items[index]  # Get the item name using the index
            normalized_item_score = normalized_scores[index]  # Get the item score using the index
            print("                  %s, Score: %.3f" % (item_name, normalized_item_score))


# %%
def get_recommendations(model, user, items, user_to_product_interaction_matrix, user2index_map, product_to_feature_interaction_matrix, user_features_csr):

    # Getting the userindex
    userindex = user2index_map.get(user, None)
    if userindex is None:
        return None

    # Preparing the user feature for the specific user
    single_user_feature = user_features_csr[userindex]

    # Products already bought
    known_positives = items[user_to_product_interaction_matrix.tocsr()[userindex].indices]
    print('User index =', userindex)

    # Scores from model prediction
    # Now including both item_features and user_features in the predict call
    scores = model.predict(
        user_ids=userindex,
        item_ids=np.arange(user_to_product_interaction_matrix.shape[1]),
        item_features=product_to_feature_interaction_matrix,
        user_features=single_user_feature  # Include user features here
    )

    # Normalize the scores to be between 0 and 1
    min_score = np.min(scores)
    max_score = np.max(scores)
    normalized_scores = (scores - min_score) / (max_score - min_score) if max_score != min_score else np.zeros_like(scores)

    # Sort the scores and get the indices of the sorted scores
    sorted_indices = np.argsort(-normalized_scores)

    # Printing out the result
    print("User %s" % user)
    print("     Known positives:")  # already known products
    for x in known_positives[:10]:
        print("                  %s" % x)

    print("     Recommended:")  # products that are recommended to the user
    for index in sorted_indices[:10]:  # Iterate over the indices of the top items
        item_name = items[index]  # Get the item name using the index
        normalized_item_score = normalized_scores[index]  # Get the item score using the index
        print("                  %s, Score: %.3f" % (item_name, normalized_item_score))


# %%
# check for the reccomendation
get_recommendations(final_model,17017,items,user_to_product_interaction,user_to_index_mapping,combined_item_features, user_features_csr)

# %%
get_recommendations(final_model,18287,items,user_to_product_interaction,user_to_index_mapping,combined_item_features)

# %%
get_recommendations(final_model,13933,items,user_to_product_interaction,user_to_index_mapping,combined_item_features)

# %%



