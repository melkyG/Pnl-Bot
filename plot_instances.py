import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from research_tool2 import find_price_velocity

# Define the file path and the desired Unix timestamp range
file_path = 'btcpricehistorydataold.hdf5'
candle_val = 1 # open = 1, close = 4
midstamp = 1465552000
width = 4000000
# Cluster Vars
step_size = 2  # Adjust this based on the desired step size
percent = 0.005  # Desired percentage difference

#User inputs
price_change = 15,15 #30  # Percentage
#or_greater = 'true'
direction = 'up'
scan_range = (1505974400, 1884000000)#(1664793600, 1684060800)  # Example scan time range
time_change = 86400*40#   # 2 days in seconds
timeframe = "1-day"

tf = timeframe

#indicators
ma200 = "true"
clusters = "false"
crossovers = "false"


midstamps = find_price_velocity(price_change, scan_range, time_change, timeframe, direction)


with h5py.File(file_path, 'r') as f:
    dset = f["1-minute/ohlc"]
    latest_unix = dset[-1][0]

for row in midstamps:
    midstamp = row
    # Load the OHLC data from the HDF5 file
    with h5py.File(file_path, 'r') as f:
        identified_windows = []  # Initialize the list to store the identified windows
        identified_windows2 = []
        crossing_points_above = []
        crossing_points_below = []
        filtered_rows = []
        window_size = 100  # Define the size of the rolling window
        middle_region_size = 10  # Define the size of the middle region
        ohlc_group = f[f'{tf}/ohlc']
        timestamps = ohlc_group[:, 0]
        timestamp_indices = (timestamps >= midstamp-width) & (timestamps <= midstamp+time_change+width)
        ohlc_data = ohlc_group[timestamp_indices]
        
        row_indices = np.where(timestamp_indices)[0]
        total_rows = len(timestamps)
        start = row_indices[0] - total_rows
        end = row_indices[-1] - total_rows
        
        # Read the datasets
        ma_200 = f[f'{tf}/200ma'][start:end, 1]  # 200-day Moving Average
        ema_200 = f[f'{tf}/200ema'][start:end, 1]  # 200-day Exponential Moving Average
        close_price = f[f'{tf}/ohlc'][start:end, candle_val]  # Close price
        # Convert unix timestamps to datetime
        timestampz = f[f'{tf}/ohlc'][start:end, 0]
        timestamps_list = timestampz.tolist()         
            
    # Plot the closing price chart
    timestamps = ohlc_data[1:, 0]
    prices = ohlc_data[1:, candle_val]
    plt.xlabel('Unix Timestamp')
    plt.ylabel('Closing Price')
    plt.title('Merry Christmas')  
    #plt.yscale('log')

    # Plot the data
    plt.plot(timestamps, prices, linewidth=0.4, color='black')#, alpha=0, ma, ema)# )
    
    plt.tight_layout()
    plt.axvline(x=midstamp, color='blue', linestyle='--', linewidth=0.5)
    plt.axvline(x=midstamp+time_change, color='blue', linestyle='--', linewidth=0.5)
    
    #moving averages
    if ma200 == 'true':
        plt.plot(timestamps, ma_200, label='200MA', color='#7A00A1', linewidth=0.3)
        plt.plot(timestamps, ema_200, label='200EMA', color='#CFA6FF', linewidth=0.3)
        # Fill the area between the 200MA and 200EMA
        plt.fill_between(timestamps, ma_200, ema_200, where=ema_200 >= ma_200, interpolate=True, color='lightgreen', alpha=0.15)
        plt.fill_between(timestamps, ma_200, ema_200, where=ema_200 < ma_200, interpolate=True, color='lightcoral', alpha=0.15)    

    #crossovers
    if crossovers == 'true':
        # Find the rows where the close price is either highest or lowest
        for i in range(len(close_price)-1):
            current_close_price = close_price[i]
            current_ma_value = ma_200[i]
            current_ema_value = ema_200[i]
            
            # Check if the close price is either the highest or lowest among the three values
            if current_close_price > max(current_ma_value, current_ema_value) or \
                current_close_price < min(current_ma_value, current_ema_value):
                filtered_rows.append(i)
        
        # Find the crossing points
        for i in range(1, len(filtered_rows)):
            prev_row_index = filtered_rows[i-1]
            current_row_index = filtered_rows[i]
            prev_close_price = close_price[prev_row_index]
            current_close_price = close_price[current_row_index]
                
            if prev_close_price < ma_200[prev_row_index] and prev_close_price < ema_200[prev_row_index] and \
                current_close_price > ma_200[current_row_index] and current_close_price > ema_200[current_row_index]:
                crossing_points_above.append(timestampz[current_row_index])
                    
            if prev_close_price > ma_200[prev_row_index] and prev_close_price > ema_200[prev_row_index] and \
                current_close_price < ma_200[current_row_index] and current_close_price < ema_200[current_row_index]:
                crossing_points_below.append(timestampz[current_row_index])
        # Plot vertical lines for crossing above points (green)
        for point in crossing_points_above:
            plt.axvline(x=point, color='green', linestyle='--', linewidth=0.5)   
        # Plot vertical lines for crossing below points (red)
        for point in crossing_points_below:
            plt.axvline(x=point, color='red', linestyle='--', linewidth=0.5)
    
    #clusters
    if clusters == 'true':
        # Iterate over the dataset using a rolling window
        for i in range(0, len(ohlc_data) - window_size + 1, step_size):
            window = ohlc_data[i: i + window_size]  # Get the current window
            
            middle_region = window[45:55]  # Extract the middle region of the window
            left_outer_region = window[:9]  # Extract the left outer region of the window
            right_outer_region = window[-9:]  # Extract the right outer region of the window
            
            # Calculate the average high value of the middle region
            middle_avg_high = np.mean(middle_region[:, 2])  # Column index 2 for the high values
            
            # Calculate the average low value of the left and right outer regions
            left_outer_avg_low = np.mean(left_outer_region[:, 3])  # Column index 3 for the low values
            right_outer_avg_low = np.mean(right_outer_region[:, 3])  # Column index 3 for the low values
            
            # Calculate the threshold value
            threshold1 = left_outer_avg_low * (1 + percent)  # Desired percentage difference
            threshold2 = right_outer_avg_low * (1 + percent)  # Desired percentage difference
            
            # Check if the criteria are met
            if middle_avg_high > threshold1 and middle_avg_high > threshold2:
                unix_time = middle_region[np.argmax(middle_region[:, 2]), 0]  # Extract the Unix time of the element with the highest high value
                high_value = np.max(middle_region[:, 2])  # Column index 2 for the high values
                
                # Append the information to the identified windows list
                identified_windows.append((unix_time, high_value))
                
        # Iterate over the dataset using a rolling window
        for i in range(0, len(ohlc_data) - window_size + 1, step_size):
            window2 = ohlc_data[i: i + window_size]  # Get the current window      
            
            middle_region = window2[45:55]  # Extract the middle region of the window
            left_outer_region = window2[:9]  # Extract the left outer region of the window
            right_outer_region = window2[-9:]  # Extract the right outer region of the window
            
            # Calculate the average low value of the middle region
            middle_avg_low = np.mean(middle_region[:, 3])  # Column index 3 for the low values
            
            # Calculate the average high value of the left and right outer regions
            left_outer_avg_high = np.mean(left_outer_region[:, 2])  # Column index 2 for the high values
            right_outer_avg_high = np.mean(right_outer_region[:, 2])  # Column index 2 for the high values
            
            # Calculate the threshold value
            threshold1 = left_outer_avg_high * (1 - percent)  # Desired percentage difference
            threshold2 = right_outer_avg_high * (1 - percent)  # Desired percentage difference
                
            # Check if the criteria are met
            if middle_avg_low < threshold1 and middle_avg_low < threshold2:
                unix_time = middle_region[np.argmin(middle_region[:, 3]), 0]  # Extract the Unix time of the element with the lowest low value
                low_value = np.min(middle_region[:, 3])  # Column index 3 for the low values
                
                # Append the information to the identified windows list
                identified_windows2.append((unix_time, low_value))
        # Add arrows pointing down for the identified windows
        for window in identified_windows:
            unix_timestamp = window[0]
            highest_high_value = window[1]
            index = np.where(timestamps == unix_timestamp)[0][0]
            price = prices[index]
            arrow_height = highest_high_value   # Adjust the value as per your preference
            plt.annotate('', xy=(unix_timestamp, price), xytext=(unix_timestamp, arrow_height),
                         arrowprops=dict(facecolor='red', arrowstyle='simple'))
            
        # Add arrows pointing down for the identified windows
        for window2 in identified_windows2:
            unix_timestamp = window2[0]
            lowest_low_value = window2[1]
            index = np.where(timestamps == unix_timestamp)[0][0]
            price = prices[index]
            arrow_height = lowest_low_value  # Adjust the value as per your preference
            plt.annotate('', xy=(unix_timestamp, price), xytext=(unix_timestamp, arrow_height),
                         arrowprops=dict(facecolor='green', arrowstyle='simple'))
    
    plt.show()
    #print(num)
