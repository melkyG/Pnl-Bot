import h5py
import matplotlib.pyplot as plt
import numpy as np


class Timeframe:
    def __init__(self, ohlc, unix, ma, ema, tf):
        self.ohlc = ohlc
        self.unix = unix
        self.ma = ma
        self.ema = ema
        self.tf = tf
        self.bias = None
    
    def set_bias(self, bias):
        self.bias = bias
    def set_tf(self, tf):
        self.tf = tf

def calculate_bias2(timeframes, unix_range):
    with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
        biggest_timeframe = None
        max_unix_gap = 0
        
        # Get the Unix values and find the biggest timeframe
        biggest_timeframe = max(timeframes, key=lambda tf: f[f'{tf}/ohlc'][-1, 0] - f[f'{tf}/ohlc'][-2, 0])
        unix_biggest = f[f'{biggest_timeframe}/ohlc'][:, 0]
        
        # Filter the Unix values based on the given range
        mask = (unix_biggest >= unix_range[0]) & (unix_biggest <= unix_range[1])
        filtered_unix_biggest = unix_biggest[mask]
        timeframe_objects = []
        for stamp in filtered_unix_biggest:
            #print(stamp)
            # Initialize the object for each timeframe
            for timeframe in timeframes:
                #print(timeframe)
                # Get the specific rows that match the stamps in filtered_unix_biggest
                
                unix_values = f[f'{timeframe}/ohlc'][:, 0]
                
                rows_to_load = np.isin(unix_values, filtered_unix_biggest)
                filtered_unix_values = unix_values[rows_to_load]
        
                ohlc_values = f[f'{timeframe}/ohlc'][rows_to_load, 4]
                ma_values = f[f'{timeframe}/200ma'][rows_to_load, 1]
                ema_values = f[f'{timeframe}/200ema'][rows_to_load, 1]
        
                # Iterate over the loaded rows
                for i, stamp in enumerate(filtered_unix_values):
                    ohlc = ohlc_values[i]
                    ma = ma_values[i]
                    ema = ema_values[i]
        
                    # Create the Timeframe object
                    tf_obj = Timeframe(ohlc, stamp, ma, ema, timeframe)
                    #zprint("created", timeframe)
                    tf_obj.set_bias('Neutral')
                    tf_obj.set_tf(timeframe)
        
                    # Set the bias based on the values
                    if ohlc == np.max([ohlc, ma, ema]):
                        tf_obj.set_bias('Bullish')
                    elif ohlc == np.min([ohlc, ma, ema]):
                        tf_obj.set_bias('Bearish')
        
                    # Append the object to the list
                    timeframe_objects.append(tf_obj)   
                    
        count = 0
        for timestamp in unix_biggest:
            if (1741564800-100*(0.5*86400)) <= timestamp <= 1741564800+10*(0.5*86400):
                count += len(timeframes)
                
        print(count)

        
        return timeframe_objects, biggest_timeframe

timeframes = ["1-day"]#, "15-minute"]#, "12-hour", "6-hour", "4-hour", "2-hour", "1-hour", "30-minute", "15-minute", "5-minute" 
unix_range = (1741564800-100*(0.5*86400),1741564800+10*(0.5*86400))#(1728432000, 1728432000 + (2*60*60*4))#(1734307200 - (8000*8400), 1734307200)#((1536054400 + 8400000), 1720224000 - 84000)  # Example ]Unix range
timeframe_objects, biggest_timeframe = calculate_bias2(timeframes, unix_range)
#print(len(objects))
# Accessing the bias attribute of each object
#for obj in objects:
#    print(obj.bias)
with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
    unix_biggest = f[f'{biggest_timeframe}/ohlc'][:, 0]
timestamps_no_bearish = []
timestamps_no_bullish = []

# Iterate over each unique unix timestamp
for timestamp in set(obj.unix for obj in timeframe_objects):
    # Initialize flags for checking bearish and bullish bias
    has_bearish = False
    has_bullish = False
    
    # Iterate over each timeframe
    for timeframe in timeframes:
        # Check if there is a Timeframe object with the same unix timestamp and timeframe
        matching_objects = [obj for obj in timeframe_objects if obj.unix == timestamp and obj.tf == timeframe]
        
        if matching_objects:
            # Check if any of the matching objects have bearish or bullish bias
            if any(obj.bias == 'Bearish' for obj in matching_objects):
                has_bearish = True
            if any(obj.bias == 'Bullish' for obj in matching_objects):
                has_bullish = True
    
    # If there is no bearish bias for the timestamp in any timeframe, add it to the list
    if not has_bearish:
        timestamps_no_bearish.append(timestamp)
    
    # If there is no bullish bias for the timestamp in any timeframe, add it to the list
    if not has_bullish:
        timestamps_no_bullish.append(timestamp)
print(timestamps_no_bearish, timestamps_no_bullish)

with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
    ohlc_values = f[f'{biggest_timeframe}/ohlc'][:, 4]
    unix_values = f[f'{biggest_timeframe}/ohlc'][:, 0]
    ma_values = f[f'{biggest_timeframe}/200ma'][:, 1]
    ema_values = f[f'{biggest_timeframe}/200ema'][:, 1]

# Filter the data within the specified Unix range
mask = (unix_values >= unix_range[0]) & (unix_values <= unix_range[1])
filtered_unix_values = unix_values[mask]
filtered_ohlc_values = ohlc_values[mask]
filtered_ma_values = ma_values[mask]
filtered_ema_values = ema_values[mask]

# Plot the filtered data
plt.plot(filtered_unix_values, filtered_ohlc_values, label='Price', color='black', markersize=0.6, linestyle=':')
plt.plot(filtered_unix_values, filtered_ma_values, label='200MA', color='#7A00A1', linewidth=0.3)
plt.plot(filtered_unix_values, filtered_ema_values, label='200EMA', color='#CFA6FF', linewidth=0.3)
# Fill between MA and EMA
plt.fill_between(filtered_unix_values, filtered_ma_values, filtered_ema_values, where=filtered_ema_values >= filtered_ma_values, interpolate=True, color='lightgreen', alpha=0.15)
plt.fill_between(filtered_unix_values, filtered_ma_values, filtered_ema_values, where=filtered_ema_values < filtered_ma_values, interpolate=True, color='lightcoral', alpha=0.15)

# Add vertical lines for timestamps with no bearish and bullish bias
for timestamp in timestamps_no_bearish:
    if unix_range[0] <= timestamp <= unix_range[1]:
        plt.axvline(x=timestamp, color='green', linestyle='-', linewidth=0.4, alpha=0.25)
for timestamp in timestamps_no_bullish:
    if unix_range[0] <= timestamp <= unix_range[1]:
        plt.axvline(x=timestamp, color='red', linestyle='-', linewidth=0.4, alpha=0.25)

# Set labels and legend
plt.xlabel('Unix Timestamp')
plt.ylabel('Value')
#plt.legend()
#plt.yscale('log')
#plt.savefig('plot2.png', dpi=300)

# Show the plot
plt.show()