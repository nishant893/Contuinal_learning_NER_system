import pandas as pd

def combine_data_with_samples(main_data, sample_data_list, sample_size=100):
    combined_data = main_data.copy()
    for sample_data in sample_data_list:
        random_samples = sample_data.sample(n=sample_size, random_state=0)
        combined_data = pd.concat([combined_data, random_samples])
    return combined_data
