import pandas as pd
import pprint
import os

# Load pickle
# obj = pd.read_pickle('./data/raw/DL_dataset.pkl')
obj_news = pd.read_pickle('./data/trainable/final_data.pkl')
print(obj_news['AAPL'].head())
print(obj_news['AAPL'].index.date())

def get_dict_structure(data):
    """
    Recursively extracts the structure (keys and value types) of a dictionary,
    handling nested dictionaries.
    """
    # Check if the input is a dictionary
    if not isinstance(data, dict):
        # If it's not a dict (e.g., a list, string, int, etc.), return its type name
        return type(data).__name__

    # If it is a dictionary, initialize an empty dictionary to hold the structure
    structure = {}
    
    # Iterate over the key-value pairs of the input dictionary
    for key, value in data.items():
        # Check the type of the current value
        if isinstance(value, dict):
            # If the value is a nested dictionary, recursively call the function
            structure[key] = get_dict_structure(value)
        elif isinstance(value, list):
            # If the value is a list, we need to handle its potential contents.
            # We'll check the first element (if the list is not empty) 
            # to represent the type of elements in the list.
            if value:
                # Recursively check the structure of the first element.
                # The structure for the list will be a list containing 
                # the structure/type of its first element.
                # Note: This assumes all elements in the list have the same structure/type.
                element_structure = get_dict_structure(value[0])
                structure[key] = [element_structure]
            else:
                # For an empty list, just record it as 'list'
                structure[key] = 'list'
        else:
            # For any other type (int, str, bool, float, etc.), just record the type name
            structure[key] = type(value).__name__
            
    return structure


# print(get_dict_structure(obj['B19ST9-R']))

# print(obj['B19ST9-R']['earnings']['Earnings Date'].head())

# print(obj['B19ST9-R']['news']['event_type'])
# print(obj['B19ST9-R']['news']['event_time'])
# print(obj['B19ST9-R']['news']['event_data'])
# print(obj['B19ST9-R']['fundamentals']['net_debt'])


# ticker = yf.Ticker("AAPL")
# earning_calendar = ticker.calendar
# print(earning_calendar)

# url = "https://www.nasdaq.com/market-activity/earnings"
# tables = pd.read_html(url)
# print(tables[0])