import pandas as pd
import numpy as np
import os
import random

def process_data():
    # Define paths
    base_dir = './OOD_Data/epinions'
    file_name = 'epinions.inter'
    file_path = os.path.join(base_dir, file_name)
    
    # Check if file exists, fallback to epinions.inter if not
    if not os.path.exists(file_path):
        alt_file = 'epinions.inter'
        alt_path = os.path.join(base_dir, alt_file)
        if os.path.exists(alt_path):
            print(f"Note: '{file_name}' not found. Using '{alt_file}' instead.")
            file_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find '{file_name}' or '{alt_file}' in '{base_dir}'.")

    print(f"Processing file: {file_path}")

    # 1. Load Data
    # The file is likely tab-separated based on extension and common RecBole format
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Clean column names (remove types like :token, :float)
    df.columns = [col.split(':')[0] for col in df.columns]
    
    required_columns = ['user_id', 'item_id']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: Missing required columns. Found: {df.columns}")
        return

    print(f"Original interactions: {len(df)}")
    print(f"Original users: {df['user_id'].nunique()}")
    print(f"Original items: {df['item_id'].nunique()}")

    # 2. Filter items with < 10 interactions
    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= 10].index
    df_filtered = df[df['item_id'].isin(valid_items)].copy()
    
    print(f"Interactions after filtering: {len(df_filtered)}")
    print(f"Items after filtering: {df_filtered['item_id'].nunique()}")

    # 3. Re-encode user_id and item_id to be continuous
    # Sort unique IDs to ensure deterministic mapping
    unique_users = sorted(df_filtered['user_id'].unique())
    unique_items = sorted(df_filtered['item_id'].unique())
    
    user_map = {u: i + 1 for i, u in enumerate(unique_users)} # Starting from 1 based on example?
    # The example shows IDs like 72, 116. If they are re-encoded continuously, they should be 1, 2, 3...
    # But the user said "Attempt to re-encode ... to ensure continuous numbering".
    # If I re-encode, user IDs will be 1..N.
    # The example has `72`, `116` which suggests maybe they are NOT re-encoded to 1..N in the example, 
    # OR the example is just an example of FORMAT.
    # Given "re-encode ... continuous", I will map to 1..N (or 0..N-1).
    # RecBole usually uses 1-based for IDs (0 for padding).
    # The example has `1` in the second column (first item).
    # I will use 1-based indexing for both users and items.
    
    item_map = {i: j + 1 for j, i in enumerate(unique_items)}
    
    df_filtered['user_id'] = df_filtered['user_id'].map(user_map)
    df_filtered['item_id'] = df_filtered['item_id'].map(item_map)
    
    print("Re-encoding complete.")
    
    # 4. Fair Dataset Splitting
    # "Ensure in test set each item appears 4-5 times"
    # "Train and test disjoint"
    # User constraint: Ensure all users in test set also appear in train set to avoid empty lists in evaluation.
    
    # We iterate through each item and pick 4 or 5 interactions.
    
    random.seed(2024)
    np.random.seed(2024)
    
    # Group interactions by item
    item_groups = df_filtered.groupby('item_id')
    
    # Group interactions by user to track user counts
    user_counts = df_filtered['user_id'].value_counts()
    
    test_indices = []
    
    for item_id, group in item_groups:
        indices = group.index.values
        count = len(indices)
        
        # Determine number of test samples (4 or 5)
        n_test = int(np.random.choice([4, 5]))
        
        # Safety check for small item counts (though filtered >= 10)
        if count < n_test:
            n_test = count
            
        # Candidate selection logic to respect user constraint
        # We prefer picking interactions where the user has > 1 interactions total
        # so that if we pick one for test, at least one remains for train.
        
        candidate_indices = []
        fallback_indices = []
        
        for idx in indices:
            uid = df_filtered.loc[idx, 'user_id']
            if user_counts[uid] > 1:
                candidate_indices.append(idx)
            else:
                fallback_indices.append(idx)
                
        # Try to sample from safe candidates first
        selected = []
        
        if len(candidate_indices) >= n_test:
            selected = np.random.choice(candidate_indices, size=n_test, replace=False)
        else:
            # Take all safe candidates and fill rest from fallback
            selected.extend(candidate_indices)
            n_needed = n_test - len(candidate_indices)
            if len(fallback_indices) >= n_needed:
                fillers = np.random.choice(fallback_indices, size=n_needed, replace=False)
                selected.extend(fillers)
            else:
                # Take everything if not enough (should rarely happen given item count >=10)
                selected.extend(fallback_indices)
        
        test_indices.extend(selected)
        
    test_indices = set(test_indices)
    
    # Final check: Ensure every user in test set has at least one interaction in train set
    # If not, move one interaction from test back to train
    
    # Construct preliminary split
    df_filtered['is_test'] = df_filtered.index.isin(test_indices)
    
    # Identify users who are in test but NOT in train
    # These are users where all their interactions ended up in test_indices
    # Or users who only had 1 interaction total and it got picked for test
    
    # Fast check using sets
    all_users = set(df_filtered['user_id'].unique())
    train_users_pre = set(df_filtered[~df_filtered['is_test']]['user_id'].unique())
    test_users_pre = set(df_filtered[df_filtered['is_test']]['user_id'].unique())
    
    # Users present in test but missing from train
    problematic_users = test_users_pre - train_users_pre
    
    print(f"Initial split check - Problematic users (in test but not train): {len(problematic_users)}")
    
    # Fix problematic users: Move one interaction from test to train for each
    if len(problematic_users) > 0:
        # Get all test interactions for these users
        prob_mask = (df_filtered['is_test']) & (df_filtered['user_id'].isin(problematic_users))
        prob_df = df_filtered[prob_mask]
        
        # For each problematic user, pick one interaction index to flip is_test=False
        # We group by user and take the first one (or random)
        
        # Taking the first interaction index for each user to remove from test set
        indices_to_flip = prob_df.groupby('user_id').head(1).index.values
        
        # Remove these from test_indices
        test_indices = test_indices - set(indices_to_flip)
        
        # Update mask
        df_filtered['is_test'] = df_filtered.index.isin(test_indices)
        
        # Re-verify
        train_users_post = set(df_filtered[~df_filtered['is_test']]['user_id'].unique())
        test_users_post = set(df_filtered[df_filtered['is_test']]['user_id'].unique())
        remaining_problems = test_users_post - train_users_post
        print(f"After fix - Problematic users: {len(remaining_problems)}")

    train_df = df_filtered[~df_filtered['is_test']]
    test_df = df_filtered[df_filtered['is_test']]
    
    print(f"Train interactions: {len(train_df)}")
    print(f"Test interactions: {len(test_df)}")
    
    # Verify coverage in test
    test_item_counts = test_df['item_id'].value_counts()
    print(f"Test item coverage stats:\n{test_item_counts.describe()}")
    
    # 5. Write Output Files
    # Format: user_id item_id1 item_id2 ...
    
    def save_formatted(dataframe, filename):
        output_path = os.path.join(base_dir, filename)
        print(f"Saving {output_path}...")
        
        # Group by user and collect items
        user_items = dataframe.groupby('user_id')['item_id'].apply(list)
        
        with open(output_path, 'w') as f:
            for user_id, items in user_items.items():
                # Sort items for consistency
                sorted_items = sorted(items)
                items_str = ' '.join(map(str, sorted_items))
                f.write(f"{user_id} {items_str}\n")
                
    save_formatted(train_df, 'train.txt')
    save_formatted(test_df, 'test.txt')
    
    print("Conversion finished successfully.")

if __name__ == "__main__":
    process_data()
