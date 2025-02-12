import pandas as pd

def clean_columns(df):
    # Keep only letters and numbers, remove everything else
    df['cleaned_Manufacturer Name'] = df['Manufacturer Name'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    df['cleaned_Manufacturer PID'] = df['Manufacturer PID'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    df['cleaned_model'] = df['model'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    df['match_Manufacturer Name_cleaned'] = df['matched_Manufacturer Name'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    df['match_Manufacturer PID_cleaned'] = df['matched_Manufacturer PID'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
    df['match_model_cleaned'] = df['matched_model'].str.lower().str.replace(r'[^a-z0-9]', '', regex=True)

    return df


def calculate_new_score(row):
    score = 100  
    
    # Define the columns to compare
    columns_to_compare = [('cleaned_Manufacturer_Name', 'match_Manufacturer_Name_cleaned'),
        ('cleaned_Manufacturer_PID', 'match_Manufacturer_PID_cleaned'),
        ('cleaned_model', 'match_model_cleaned')]

    # Compare ignoring NaN
    for col1, col2 in columns_to_compare:
        if pd.isna(row[col1]) or pd.isna(row[col2]):
            continue
        
        if row[col1].lower() != row[col2].lower():  # Reduce the score, in case no matching
            score -= 33
    
    return score


def calculate_pid_brand_score(row):
    # Initialize the score to 0
    score = 0

    # Matches in PID
    pid_match = (
        not pd.isna(row['matched_pid_left']) and not pd.isna(row['matched_pid_right']) and
        row['matched_pid_left'].lower() == row['matched_pid_right'].lower()
    )
    
    # Mtches in Brand
    brand_match = (
        not pd.isna(row['matched_brand_left']) and not pd.isna(row['matched_brand_right']) and
        row['matched_brand_left'].lower() == row['matched_brand_right'].lower()
    )

    # Scores
    if pid_match and brand_match: 
        score = 100
    elif pid_match or brand_match: 
        score = 50
    else:  
        score = 0

    return score