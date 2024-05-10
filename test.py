import pandas as pd

# Assuming 'raw_data' is your DataFrame
# Create some sample data for demonstration
data = {'viral': [0, 1, 0, 1, 0],
        'item_author_cate_index': [1, 2, 1, 3, 2],
        'article_author_index': [4, 5, 4, 6, 5],
        'article_source_cate_index': [7, 8, 7, 9, 8],
        'other_column': ['A', 'B', 'C', 'D', 'E']}
raw_data = pd.DataFrame(data)

# Find all rows where 'viral' column has a value of 1
positive_rows = raw_data[raw_data['viral'] == 1]

# Initialize an empty list to store concatenated rows
concatenated_rows = []

# Iterate over each positive row
for index, positive_row in positive_rows.iterrows():
    # Find corresponding negative row based on specified conditions
    negative_rows = raw_data[(raw_data['item_author_cate_index'] == positive_row['item_author_cate_index']) &
                             (raw_data['article_author_index'] == positive_row['article_author_index']) &
                             (raw_data['article_source_cate_index'] == positive_row['article_source_cate_index']) &
                             (raw_data['viral'] == 0)]
    
    # Check if there are valid negative rows
    if not negative_rows.empty:
        # Take the first negative row
        negative_row = negative_rows.iloc[0]
        
        # Concatenate negative row to positive row with modifications
        concatenated_row = pd.concat([positive_row, negative_row.add_prefix('neg_')])
        
        # Append concatenated row to the list
        concatenated_rows.append(concatenated_row)

# Create a new DataFrame from the list of concatenated rows
result_df = pd.DataFrame(concatenated_rows)

print(result_df)
