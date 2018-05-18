
# coding: utf-8

# In[1]:


import pandas as pd
cars = pd.read_csv("imports-85.data", 
                   names = ['symboling', 'normalized_losses', 
                              'make', 'fuel_type', 'aspiration',
                             'num_doors', 'body_style', 'drive_wheels',
                             'engine_location', 'wheel_base', 'length',
                             'width', 'height', 'curb_weight', 'engine_type',
                             'num_cylinders', 'engine_size', 'fuel_system',
                             'bore', 'stroke', 'compression_ratio', 
                              'horsepower', 'peak_rpm', 'city_mpg',
                              'highway_mpg', 'price'])


# In[2]:


cars.head(10)


# Target column = price
# Numeric columns = normalised_losses, wheel_base, length, width, height, curb-weight, engine_size, bore, stroke, compression-ratio, horsepower, peak-rpm, city-mpg, highway-mpg
# Feature columns = symboling, make, fuel_type, aspiration, num-of-doors, body-style, drive-wheels, engine-location, engine-type, num-of-cylinders, fuel-system

# In[3]:


import numpy as np
cars = cars.replace("?", np.nan)


# In[4]:


cars.dtypes


# In[5]:


cars['normalized_losses'] = cars['normalized_losses'].astype(float)
cars['bore'] = cars['bore'].astype(float)
cars['stroke'] = cars['stroke'].astype(float)
cars['horsepower'] = cars['horsepower'].astype(float)
cars['peak_rpm'] = cars['peak_rpm'].astype(float)
cars['price'] = cars['price'].astype(float)


# In[6]:


cars['normalized_losses'].isnull().sum()


# In[7]:


cars['normalized_losses'].value_counts()


# In[8]:


cars['normalized_losses'].mean()


# In[9]:


cars['normalized_losses'].shape[0]


# Normalised losses:
# total rows = 205
# mean = 122 (4 occurences)
# Mode = 161 (11 occurences)
# empty values = 41
# 
# Look at what happens after removing rows with empty price first, maybe that will solve the problem

# In[10]:


cars = cars.dropna(subset=['price'])
cars.isnull().sum()


# In[11]:


cars = cars.dropna(subset=['price', 'num_doors', 'bore', 'stroke',
                          'horsepower', 'peak_rpm'])
cars.isnull().sum()


# In[12]:


cars.shape[0]


# In[13]:


34/193


# In[14]:


cars['normalized_losses'].mean()


# after visually inspecting the data, values for normalised losses are quite similar within makes. Take averages within and use assign

# In[15]:


cars['make'].value_counts()


# In[16]:


makes = ['toyota', 'nissan', 'mitsubishi', 'honda', 'volkswagen',
        'subaru', 'mazda', 'peugot', 'volvo', 'bmw', 'mercedes-benz',
        'dodge', 'plymouth', 'saab', 'audi', 'porsche', 'jaguar',
        'chevrolet', 'alfa-romero', 'isuzu', 'mercury']
make_averages = []
for make in makes:
    cars_make = cars[cars['make'] == make]
    make_average = cars_make['normalized_losses'].mean()
    report = make, " average is ", make_average
    make_averages.append(report)
    
make_averages


# for a bigger dataset, would deal with quicker but for here, can inspect and find 0 values with common characteristics

# In[17]:


cars['norm_losses'] = cars['normalized_losses']

if cars.loc[(cars['normalized_losses'] == 0) &
            (cars['make'] == 'toyota')] is True:
    cars['norm_losses'] = 91


# In[18]:


cars.tail(100)


# In[19]:


cars.set_value(181, 'normalized_losses', 91)


# In[20]:


cars.isnull().sum()


# In[21]:


pd.set_option('display.max_rows', 210)


# In[22]:


cars


# In[23]:


#Audi - mean
cars.set_value(5, 'normalized_losses', 161)
cars.set_value(7, 'normalized_losses', 161)
#BMW - mean
cars.set_value(14, 'normalized_losses', 190)
cars.set_value(15, 'normalized_losses', 190)
cars.set_value(16, 'normalized_losses', 190)
cars.set_value(17, 'normalized_losses', 190)
#Isuzu - empty so use Honda average
cars.set_value(43, 'normalized_losses', 103)
cars.set_value(46, 'normalized_losses', 103)
#Jaguar - mean
cars.set_value(48, 'normalized_losses', 145)
cars.set_value(49, 'normalized_losses', 145)
#Mazda - based on average of other Mazda Sedans
cars.set_value(66, 'normalized_losses', 115)
#Mercedes-Benz - based on other MB gas cars
cars.set_value(71, 'normalized_losses', 142)
cars.set_value(73, 'normalized_losses', 142)
cars.set_value(74, 'normalized_losses', 142)

#Leave Mercury and just delete row after

#Mitsubishi - mean
cars.set_value(82, 'normalized_losses', 146)
cars.set_value(83, 'normalized_losses', 146)
cars.set_value(84, 'normalized_losses', 146)
#Peugeot - only 161
cars.set_value(113, 'normalized_losses', 161)
cars.set_value(114, 'normalized_losses', 161)
#plymouth - go same as other plymouth hatchback
cars.set_value(124, 'normalized_losses', 129)
#Porsche - only 186
cars.set_value(126, 'normalized_losses', 186)
cars.set_value(127, 'normalized_losses', 186)
cars.set_value(128, 'normalized_losses', 186)
#Volkswagen - one outlier excluded, mean used apart from this
cars.set_value(189, 'normalized_losses', 102)
cars.set_value(191, 'normalized_losses', 102)
cars.set_value(192, 'normalized_losses', 102)
cars.set_value(193, 'normalized_losses', 102)


# In[24]:


cars.isnull().sum()


# In[24]:


#Fix Peugeot
cars.set_value(109, 'normalized_losses', 161)
cars.set_value(110, 'normalized_losses', 161)
#Set Alfa to mean since nothing similar
cars.set_value(0, 'normalized_losses', cars['normalized_losses'].mean())
cars.set_value(1, 'normalized_losses', cars['normalized_losses'].mean())
cars.set_value(2, 'normalized_losses', cars['normalized_losses'].mean())


# In[25]:


cars.drop('norm_losses', axis=1, inplace=True)


# In[26]:


cars.isnull().sum()


# In[27]:


cars.dropna()


# In[28]:


cars.isnull().sum()


# In[29]:


cars.shape


# In[30]:


cars = cars.dropna()
cars.isnull().sum()


# Would be better to do this in more systematic way but for this dataset this was a good option

# In[31]:


# Select only the columns with continuous values from - https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized_losses', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
numeric_cars = cars[continuous_values_cols]


# In[32]:


# Normalize all columnns to range from 0 to 1 except the target column.
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min())/(numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col


# In[33]:


numeric_cars


# In[34]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

def knn_train_test(train_col, test_col, df):
    knn = KNeighborsRegressor()
    np.random.seed(1)
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(numeric_cars.index)
    rand_df = numeric_cars.reindex(shuffled_index)
    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train = df.iloc[:last_train_row]
    test = df.iloc[last_train_row:]
    # Fit a KNN model using default k value.
    knn.fit(train[[train_col]], train[test_col])
    # Make predictions using model.
    prediction = knn.predict(test[[train_col]])
    # Calculate and return RMSE.
    rmse = mean_squared_error(test[test_col], prediction)**(1/2)
    return rmse

rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.

for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val
    
# Create a Series object from the dictionary so 
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()

  
    
    


# In[35]:


#Add K value


# In[36]:


def knn_train_test(train_col, target_col, df):
    np.random.seed(1)
        
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [1,3,5,7,9]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[[train_col]], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

k_rmse_results


# In[37]:


import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')


# In[38]:



# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k,v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
series_avg_rmse.sort_values()


# In[39]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [5]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

two_best_features = ['horsepower', 'width']
rmse_val = knn_train_test(two_best_features, 'price', numeric_cars)
k_rmse_results["two best features"] = rmse_val

three_best_features = ['horsepower', 'width', 'curb_weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb_weight', 'city_mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb_weight' , 'city_mpg' , 'highway_mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

six_best_features = ['horsepower', 'width', 'curb_weight' , 'city_mpg' , 'highway_mpg', 'length']
rmse_val = knn_train_test(six_best_features, 'price', numeric_cars)
k_rmse_results["six best features"] = rmse_val

k_rmse_results


# In[41]:


def knn_train_test(train_cols, target_col, df):
    np.random.seed(1)
    
    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_df) / 2)
    
    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_df = rand_df.iloc[0:last_train_row]
    test_df = rand_df.iloc[last_train_row:]
    
    k_values = [i for i in range(1, 25)]
    k_rmses = {}
    
    for k in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(train_df[train_cols], train_df[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_df[train_cols])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_df[target_col], predicted_labels)
        rmse = np.sqrt(mse)
        
        k_rmses[k] = rmse
    return k_rmses

k_rmse_results = {}

three_best_features = ['horsepower', 'width', 'curb_weight']
rmse_val = knn_train_test(three_best_features, 'price', numeric_cars)
k_rmse_results["three best features"] = rmse_val

four_best_features = ['horsepower', 'width', 'curb_weight', 'city_mpg']
rmse_val = knn_train_test(four_best_features, 'price', numeric_cars)
k_rmse_results["four best features"] = rmse_val

five_best_features = ['horsepower', 'width', 'curb_weight' , 'city_mpg' , 'highway_mpg']
rmse_val = knn_train_test(five_best_features, 'price', numeric_cars)
k_rmse_results["five best features"] = rmse_val

k_rmse_results


# In[42]:


for k,v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    
    plt.plot(x,y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

