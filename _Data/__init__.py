import pandas as pd
import os



def get_data(file):
    """Get the data from the file
    
    Args:
        file: str, the file name

    returns:
        pd.DataFrame, the data in the file

    """
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file = file+'.csv' if not file.endswith('.csv') else file
    file = os.path.join(base_dir, file)
    if os.path.exists(file):
        return pd.read_csv(file)
    else:
        avilable_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
        error = f"The file {file} does not exist"
        error += f"Tha available files are: {avilable_files}"
        raise Exception(error)
