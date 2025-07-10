import pandas as pd


file_path = '/Users/steorra/Downloads/en.openfoodfacts.org.products.csv'

# with open('/Users/steorra/Downloads/en.openfoodfacts.org.products.csv') as f:
#     for _ in range(2):
#         print(f.readline())

cols_to_use = [
    'product_name',
    'brands',
    'main_category_en',
    'categories_en',
    'nutriscore_score',
    'nutriscore_grade',
    'nova_group',
    'additives_n',
    'additives_tags'
]

chunksize = 500_000  # Number of rows per chunk
chunks = pd.read_csv(file_path, sep='\t', usecols=cols_to_use, chunksize=chunksize, low_memory=False, on_bad_lines='skip')


# df = pd.read_csv(file_path, sep='\t', usecols=cols_to_use, low_memory=False, on_bad_lines='skip')
# df = df[df['main_category_en'].notna()]
# df.reset_index(drop=True, inplace=True)
# print(f"Filtered dataset shape: {df.shape}")
# print(df.head())
total = 0
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i + 1}")
    chunk = chunk[chunk['main_category_en'].notna()]
    chunk.reset_index(drop=True, inplace=True)
    total += chunk.shape[0]
    # Example: print column names or filter rows
    print(chunk.head())
print(total)

# print(df.shape)       # Print number of rows and columns
# print(df.head())      # Preview first few rows


# df = pd.read_csv('/Users/steorra/Downloads/en.openfoodfacts.org.products.csv', on_bad_lines='skip', low_memory=False)
# print(df)