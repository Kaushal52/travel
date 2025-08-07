import pandas as pd

# Path to the Excel file
file_path = "/Users/kaushalkhatu/Desktop/untitled folder/travel.xlsx"

# Read the Excel file
df = pd.read_excel(file_path)

# Display the first few rows
print(df.head())


import pandas as pd

# Load the Excel file
file_path = "/Users/kaushalkhatu/Desktop/untitled folder/travel.xlsx"
df = pd.read_excel(file_path)

# Display all column names
print("Column Names:")
print(df.columns.tolist())

# Drop the specified columns
df = df.drop(columns=['Health_Safety_Concerns'])

# Display the first few rows to confirm the columns are removed
print(df.head())

# Save the modified DataFrame as a CSV file
output_file = "/Users/kaushalkhatu/Desktop/untitled folder/travel_clean.csv"
df.to_csv(output_file, index=False)

print(f"File saved as {output_file}")

# Check for missing values in the dataset
missing_data = df.isnull().sum()
print("Missing data in each column:\n", missing_data)

# Descriptive statistics for numerical columns
numerical_summary = df.describe()
print("Descriptive statistics for numerical columns:\n", numerical_summary)

# Get unique values in categorical columns
categorical_columns = ['Destination', 'Travel_Type', 'Activities', 'Accommodation_Type', 'Cuisine',
                       'Past_Travel_History', 'Language', 'Geographical_Features', 
                       'Weather_Conditions', 'Accessibility', 'Accommodation_Options', 'Cultural_Highlights', 
                       'Local_Cuisine', 'User_Reviews', 'Special_Features', 'Tourist_Spots']
                       
for col in categorical_columns:
    print(f"Unique values in {col}:")
    print(df[col].nunique(), "\n")

# Correlation matrix for numerical columns
correlation_matrix = df.corr()
print("Correlation matrix for numerical columns:\n", correlation_matrix)

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram for numerical columns
df[['Budget_Min', 'Budget_Max', 'Cost']].hist(bins=15, figsize=(10, 6))
plt.show()

# Boxplot for numerical columns to check for outliers
sns.boxplot(data=df[['Budget_Min', 'Budget_Max', 'Cost']])
plt.show()

# Heatmap of correlations
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()

# Check for missing values
missing_data = df.isnull().sum()
print("Missing data in each column:\n", missing_data)

# Check for duplicate rows
duplicate_rows = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicate_rows)

# Remove duplicates if any
df = df.drop_duplicates()

# Check data types
print("\nData types of each column:\n", df.dtypes)

# Descriptive statistics for numerical columns
numerical_summary = df.describe()
print("\nDescriptive statistics for numerical columns:\n", numerical_summary)

# Summary of categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f"\nUnique values in {col}: {df[col].nunique()}")
    print(f"Top values in {col}: {df[col].value_counts().head()}")

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/Users/kaushalkhatu/Desktop/untitled folder/travel_clean.csv')

# Convert the Budget columns from ranges to numeric averages
def parse_budget_range(budget_range):
    try:
        # Split by '-' and calculate the average
        if isinstance(budget_range, str) and '-' in budget_range:
            min_val, max_val = budget_range.split('-')
            return (float(min_val) + float(max_val)) / 2
        return float(budget_range)
    except:
        return None

# Apply the conversion to the relevant columns
df['Budget_Min'] = df['Budget_Min'].apply(parse_budget_range)
df['Budget_Max'] = df['Budget_Max'].apply(parse_budget_range)
df['Cost'] = df['Cost'].apply(parse_budget_range)

# 1. Cost vs. Destination
avg_cost_by_destination = df.groupby('Destination')['Cost'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
avg_cost_by_destination.plot(kind='bar', color='skyblue')
plt.title('Average Cost by Destination')
plt.ylabel('Average Cost')
plt.xticks(rotation=45)
plt.show()

# 2. Budget vs. Destination
avg_budget_by_destination = df.groupby('Destination')[['Budget_Min', 'Budget_Max']].mean().sort_values(by='Budget_Min', ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Min budget plot
avg_budget_by_destination['Budget_Min'].plot(kind='bar', ax=axes[0], color='lightcoral')
axes[0].set_title('Average Minimum Budget by Destination')
axes[0].set_ylabel('Average Minimum Budget')
axes[0].set_xticklabels(avg_budget_by_destination.index, rotation=45)

# Max budget plot
avg_budget_by_destination['Budget_Max'].plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Average Maximum Budget by Destination')
axes[1].set_ylabel('Average Maximum Budget')
axes[1].set_xticklabels(avg_budget_by_destination.index, rotation=45)

plt.tight_layout()
plt.show()

# 3. Most Common Travel Types and Their Costs
avg_cost_by_travel_type = df.groupby('Travel_Type')['Cost'].mean().sort_values(ascending=False)
travel_type_count = df['Travel_Type'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Travel type count plot
travel_type_count.plot(kind='bar', ax=axes[0], color='lightblue')
axes[0].set_title('Count of Each Travel Type')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(travel_type_count.index, rotation=45)

# Avg cost by travel type plot
avg_cost_by_travel_type.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Average Cost by Travel Type')
axes[1].set_ylabel('Average Cost')
axes[1].set_xticklabels(avg_cost_by_travel_type.index, rotation=45)

plt.tight_layout()
plt.show()

# 4. Cuisine Preferences vs. Cost
avg_cost_by_cuisine = df.groupby('Cuisine')['Cost'].mean().sort_values(ascending=False)
cuisine_count = df['Cuisine'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Cuisine count plot
cuisine_count.plot(kind='bar', ax=axes[0], color='lightblue')
axes[0].set_title('Count of Each Cuisine')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(cuisine_count.index, rotation=45)

# Avg cost by cuisine plot
avg_cost_by_cuisine.plot(kind='bar', ax=axes[1], color='lightcoral')
axes[1].set_title('Average Cost by Cuisine')
axes[1].set_ylabel('Average Cost')
axes[1].set_xticklabels(avg_cost_by_cuisine.index, rotation=45)

plt.tight_layout()
plt.show()

# 5. Accommodation Type vs. Cost
avg_cost_by_accommodation = df.groupby('Accommodation_Type')['Cost'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
avg_cost_by_accommodation.plot(kind='bar', color='lightgreen')
plt.title('Average Cost by Accommodation Type')
plt.ylabel('Average Cost')
plt.xticks(rotation=45)
plt.show()

# 6. Activities vs. Cost
avg_cost_by_activities = df.groupby('Activities')['Cost'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
avg_cost_by_activities.plot(kind='bar', color='skyblue')
plt.title('Average Cost by Activities')
plt.ylabel('Average Cost')
plt.xticks(rotation=45)
plt.show()

# 7. Destination vs. User Reviews
user_reviews_by_destination = df.groupby('Destination')['User_Reviews'].value_counts().unstack().fillna(0)
plt.figure(figsize=(12, 8))
user_reviews_by_destination.plot(kind='bar', stacked=True)
plt.title('User Reviews by Destination')
plt.ylabel('Count of Reviews')
plt.xticks(rotation=45)
plt.show()

# 8. Comparison of Travel Types by Cost and Budget
avg_budget_and_cost_by_travel_type = df.groupby('Travel_Type')[['Budget_Min', 'Budget_Max', 'Cost']].mean()
avg_budget_and_cost_by_travel_type.plot(kind='bar', figsize=(14, 7))
plt.title('Average Budget and Cost by Travel Type')
plt.ylabel('Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the updated dataset
df = pd.read_csv("/Users/kaushalkhatu/Desktop/untitled folder/travel_clean.csv")

# Display basic information about the dataset
print("Dataset Info:")
print(df.info())

# Display the first few rows
print("\nFirst 5 rows:")
print(df.head())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Identify numerical columns
numerical_columns = df.select_dtypes(include='number').columns

# Boxplots to check for outliers
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[numerical_columns])
plt.title('Boxplot for Numerical Columns')
plt.show()

# Identify categorical columns
categorical_columns = df.select_dtypes(include='object').columns

# Countplot for each categorical column to visualize distribution
for col in categorical_columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.show()

# Correlation heatmap for numerical columns
plt.figure(figsize=(12, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for numerical columns
sns.pairplot(df[numerical_columns])
plt.title('Pairplot for Numerical Columns')
plt.show()

# Check missing values
plt.figure(figsize=(10, 7))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()



