import pandas as pd
import matplotlib.pyplot as plt
import h5py

"""
# Open the HDF5 file and navigate to the relevant datasets
with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
    ohlc_dataset = f['4-hour/ohlc']
    ma_dataset = f['4-hour/200ma']
    ema_dataset = f['4-hour/200ema']


    # Get the number of rows in the datasets
    num_rows = ohlc_dataset.shape[0]

    # Extract the close price, MA and EMA values for each row
    close_prices = [ohlc_dataset[i][4] for i in range(num_rows)]
    ma_values = [ma_dataset[i][1] for i in range(num_rows)]
    ema_values = [ema_dataset[i][1] for i in range(num_rows)]

    # Determine whether each row is bullish or bearish
    for i in range(num_rows):
        if close_prices[i] > ma_values[i] and close_prices[i] > ema_values[i]:
            print(f'Row {i + 1}: bullish')
            continue
        elif close_prices[i] < ma_values[i] and close_prices[i] < ema_values[i]:
            print(f'Row {i + 1}: bearish')
            continue
        else:
            print(f'Row {i + 1}: neutral')
"""


def determine_trend(price, ma, ema):
    if price > ma and price > ema:
        return "up"
    elif price < ma and price < ema:
        return "down"
    else:
        return "range"
"""
def find_crossing_points(file_path):
    # Load the HDF5 file
    hdf5_file = h5py.File(file_path, 'r')

    # Read the datasets
    ma_200 = hdf5_file['4-hour/200ma'][:, 1]  # 200-day Moving Average
    ema_200 = hdf5_file['4-hour/200ema'][:, 1]  # 200-day Exponential Moving Average
    close_price = hdf5_file['4-hour/ohlc'][:, 4]  # Close price

    # Convert unix timestamps to datetime
    timestamps = pd.to_datetime(hdf5_file['4-hour/ohlc'][:, 0], unit='s')

    # Find the indices where the price goes from below both indicators to above both indicators
    cross_above_indices = ((close_price[:-1] < ma_200[:-1]) & (close_price[1:] > ma_200[1:]) & (close_price[:-1] < ema_200[:-1]) & (close_price[1:] > ema_200[1:])).nonzero()[0]

    # Find the indices where the price goes from above both indicators to below both indicators
    cross_below_indices = ((close_price[:-1] > ma_200[:-1]) & (close_price[1:] < ma_200[1:]) & (close_price[:-1] > ema_200[:-1]) & (close_price[1:] < ema_200[1:])).nonzero()[0]

    # Get the unix timestamps for the crossing points
    crossing_points_above = timestamps[cross_above_indices + 1]
    crossing_points_below = timestamps[cross_below_indices + 1]

    # Close the HDF5 file
    hdf5_file.close()

    return crossing_points_above, crossing_points_below
"""

# Load the HDF5 file
file_path = 'btcpricehistorydataold.hdf5'
hdf5_file = h5py.File(file_path, 'r')

unix_start = -228#- 19450#-820
unix_end = -228+70#- 16960#-410
tf="1-day"

# Read the datasets
unix = hdf5_file[f"{tf}/ohlc"][:, 0] #[unix_start:unix_end:, 0]
ma_200 = hdf5_file[f"{tf}/200ma"][:, 1]  # 200-day Moving Average
ema_200 = hdf5_file[f"{tf}/200ema"][:, 1]  # 200-day Exponential Moving Average
close_price = hdf5_file[f"{tf}/ohlc"][:, 4]  # Close price

# Convert unix timestamps to datetime
timestamps = pd.to_datetime(hdf5_file[f"{tf}/ohlc"][:, 0], unit='s')

    
crossing_points_above = []
crossing_points_below = []
    
# Find the rows where the close price is either highest or lowest
filtered_rows = []
for i in range(len(close_price)):
    current_close_price = close_price[i]
    current_ma_value = ma_200[i]
    current_ema_value = ema_200[i]
    print(i+1," ", unix[i], close_price[i], ma_200[i], ema_200[i])
    
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
        crossing_points_above.append(timestamps[current_row_index])
            
    if prev_close_price > ma_200[prev_row_index] and prev_close_price > ema_200[prev_row_index] and \
        current_close_price < ma_200[current_row_index] and current_close_price < ema_200[current_row_index]:
        crossing_points_below.append(timestamps[current_row_index])
       
# Convert timestamps to a list
timestamps_list = timestamps.tolist()    
plt.figure(figsize=(60, 20), dpi=400)
# Plot the data
plt.plot(timestamps, ma_200, label='200MA', color='#7A00A1', linewidth=0.3)
plt.plot(timestamps, ema_200, label='200EMA', color='#CFA6FF', linewidth=0.3)
# Fill the area between the 200MA and 200EMA
plt.fill_between(timestamps, ma_200, ema_200, where=ema_200 >= ma_200, interpolate=True, color='lightgreen', alpha=0.15)
plt.fill_between(timestamps, ma_200, ema_200, where=ema_200 < ma_200, interpolate=True, color='lightcoral', alpha=0.15)

# Plot black dots for the closing price
plt.plot(timestamps, close_price, '-', color='black', linewidth=0.2,alpha=0.4, label='Close Price')


# Plot vertical lines for crossing above points (green)
for point in crossing_points_above:
    plt.axvline(x=point, color='green', linestyle='--', linewidth=0.4)

# Plot vertical lines for crossing below points (red)
for point in crossing_points_below:
    plt.axvline(x=point, color='red', linestyle='--', linewidth=0.4)

"""
# Plot arrows for crossing above points (green arrows pointing up)
for point in crossing_points_above:
    idx = timestamps.get_loc(point)
    plt.annotate('↑', (timestamps_list[idx], close_price[idx]), color='green', ha='center', xytext=(0.5,-10), textcoords='offset points')

# Plot arrows for crossing below points (red arrows pointing down)
for point in crossing_points_below:
    idx = timestamps.get_loc(point)
    plt.annotate('↓', (timestamps_list[idx], close_price[idx]), color='red', ha='center', xytext=(0.5,2), textcoords='offset points')
"""
    
# Set the plot title and labels
plt.title('btusd, 200 ma/ema')
plt.xlabel('Unix')
plt.ylabel('Price')

# Rotate x-axis tick labels for better readability
plt.xticks(rotation=45)

# Display the legend and plot the graph
#plt.legend()
plt.tight_layout()
plt.yscale('log')
#plt.figure(figsize=(10, 6), dpi=300)
plt.savefig('ma_trends_1d', dpi=400)
plt.show()

# Close the HDF5 file
hdf5_file.close()





#conditions 