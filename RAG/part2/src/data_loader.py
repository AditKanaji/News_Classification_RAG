import pandas as pd

def load_csv_data(filepath):
    """
    Loads the stress test CSV and cleans it for processing.
    Expects columns: 'title', 'bodyText' (and optional 'published_date', 'id', etc.)
    """
    try:
        df = pd.read_csv(filepath)
        
        # Ensure required columns exist (Updated for new schema)
        required_cols = ['title', 'bodyText'] 
        for col in required_cols:
            if col not in df.columns:
                print(f"Found columns: {list(df.columns)}")
                raise ValueError(f"Missing required column in CSV: {col}")
        
        # Fill NaNs to avoid JSON errors
        df.fillna("Unknown", inplace=True)
        
        # Convert to list of dicts for easy iteration
        return df.to_dict(orient='records')
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []