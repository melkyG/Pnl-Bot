import random
import numpy as np
import config
import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


#select timeframes
tf_used = config.all_timeframes[-1:]

candle_val = 4

#input range
scan_range = (1514595600+0*(0.5*86400),1741564800-2*(0.5*86400))#(1322395200-2*(0.5*86400),1739404800+(0*86400))#(1684800000+(47*60*60*4), 1684800000+(55*60*60*4)) #(1734537600, 1735228800)#(1325318400, 1325318400 + (100*60*60*4)) #(1684800000, 1684800000+(86400*50)) 

#dictionary to store data by timeframes
tf_data = {}    

#select rows to scan
debug_rows = 5

with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
    for tf in tf_used:
        start_time, end_time = scan_range
        print(f"user inputs: ({start_time}, {end_time}) length: ", end_time-start_time, type(start_time), type(end_time),"\n")

        #adjust input range if outside of dataset
        if scan_range[0] < f[f"{tf}/ohlc"][0][0] : start_time = f[f"{tf}/ohlc"][0][0] 
        if scan_range[1] > f[f"{tf}/ohlc"][-1][0] : end_time = f[f"{tf}/ohlc"][-1][0]
        print(f"corrected user inputs: ({start_time}, {end_time}) length: ", end_time-start_time, type(start_time), type(end_time),"\n")
        print("="*22,tf,"="*22)
        interval = config.tf_intervals[tf]  # Get the interval for the current timeframe
        start_time = int(((start_time + (interval - 1)) // interval) * interval)+interval
        end_time = (((end_time + (interval - 1)) // interval) * interval)#1739486880
        print(f"adj. unix range for processing ({start_time}, {end_time}) length: ", end_time-start_time,"\n")
        #print("interval types:",type(start_time), type(end_time), "\n")
        
        
        
        #ohlc_data = np.array(f[f"{tf}/ohlc"])
        #ma_data = np.array(f[f"{tf}/200ma"])
        #ema_data = np.array(f[f"{tf}/200ema"])
        ohlc_data = f[f"{tf}/ohlc"][:]
        ma_data = f[f"{tf}/200ma"][:]
        ema_data = f[f"{tf}/200ema"][:]
        print("of Database:")
        print(f"first unix & price {ohlc_data[0][0].astype(int)} {ohlc_data[0][4]}")
        print(f"last unix & price {ohlc_data[-1][0].astype(int)} {ohlc_data[-1][4].astype(int)}","\n")
        #print(ma_data[-1][1])
        #print(ema_data[-1][1])
        #ma_check=f[f"{tf}/200ma"][-50:][1]
        #print(tf, ma_check)
        
        if end_time - start_time == 0:
            print(end_time, start_time)
            range_ohlc = ohlc_data[(ohlc_data[:,0] == start_time)]
            range_ma = ma_data[(ma_data[:,0] == start_time)]
            range_ema = ema_data[(ema_data[:,0] == start_time)]
            
        else:
            # Filter rows within the range based on the first column (Unix timestamp)
            range_ohlc = ohlc_data[(ohlc_data[:, 0] >= start_time) & (ohlc_data[:, 0] <= end_time)]
            #print(range_ohlc.astype(int))
            range_ma = ma_data[(ma_data[:, 0] >= start_time) & (ma_data[:, 0] <= end_time)]
            range_ema = ema_data[(ema_data[:, 0] >= start_time) & (ema_data[:, 0] <= end_time)]
        print("of Filtered by corrected unix range:")
        print(f"first unix & price {range_ohlc[0][0].astype(int)} {range_ohlc[0][4]}")
        print(f"last unix & price {range_ohlc[-1][0].astype(int)} {range_ohlc[-1][4]}","\n")
        print(len(range_ohlc),len(range_ma),len(range_ema))
        #print("check: ", tf,range_ohlc[1][4].astype(int),range_ma[1].astype(int),range_ema[1].astype(int))
        
            
        # Generate all expected Unix timestamps within the range
        expected_unix = np.arange(start_time, end_time+interval, interval) #old was end_time+interval
        #print(expected_unix.astype(int), len(expected_unix))
        # Read the dataset for the current timeframe
        dset = f[f'{tf}/ohlc']
        
        # Create a dictionary of rows indexed by Unix timestamp for fast lookup
        data_dict = {row[0]: row for row in dset}
        
        # Create filtered data with consistent Unix gaps
        filtered_data = []
        for ts in expected_unix:
                
            if ts in data_dict:
                #print("data found: ", ts)
                filtered_data.append(data_dict[ts])
                #print(data_dict[ts].astype(int))
            else:
                # Append a placeholder row with Unix timestamp and `None` for other columns
                #print(f'({tf}) data not found: {ts}')
                filtered_data.append([ts] + [None] * (dset.shape[1] - 1))
            
        filtered_data = np.array(filtered_data, dtype=object)  # Allow `None` values
        print(len(filtered_data),filtered_data[-1][0])
        print(len(range_ma),range_ma[-1][0])
        ma_dict = {row[0]: row[1] for row in range_ma}  # Assuming range_ma has [timestamp, value]
        ema_dict = {row[0]: row[1] for row in range_ema}
        
        ma_aligned = [ma_dict.get(ts, None) for ts in expected_unix]
        ema_aligned = [ema_dict.get(ts, None) for ts in expected_unix]
        print(len(ma_aligned), ma_aligned[-1])
        
        # Compute the average between ma and ema for each row
        avg_mas = np.array((np.array(ma_aligned, dtype=np.float64) + np.array(ema_aligned, dtype=np.float64)) / 2, dtype=np.float64)
        # Compute the differences (slopes)
        diff_avg_mas = np.diff(avg_mas)
        diff_unix = np.diff(filtered_data[:, 0])
        slopes = diff_avg_mas / diff_unix  # Compute slopes

        # Extract specific columns (example: closing price, Unix timestamps, etc.)
        tf_data[tf] = {
            'unix': filtered_data[:, 0],
            'close': filtered_data[:, candle_val],
            'ma': np.array(ma_aligned, dtype=object),  # Ensures alignment
            'ema': np.array(ema_aligned, dtype=object),
            'avg_mas' : np.array(avg_mas, dtype=object),
            'slopes': np.array(slopes, dtype=object)
            
        }
        
        # Compute the differences (slopes)
        #diff_avg_mas = np.diff(tf_data[tf]['avg_mas'])
        #diff_unix = np.diff(tf_data[tf]['unix'])
        #slopes = diff_avg_mas / diff_unix  # Compute slopes

        print(tf_data)
        print(f"({tf}) Filtered data contains {len(tf_data[tf]['unix'])} row(s) within range: ({start_time}, {end_time}).\n")
    print("-"*50, "\n")  
            
def plot_timeframe_grid(tf_used, start_time, end_time, candle_val, config, filename='btcpricehistorydataold.hdf5'):
    # Create figure
    fig, ax = plt.subplots(figsize=(70, 4))
    
    # Process and plot each timeframe
    row_positions = {tf: idx for idx, tf in enumerate(reversed(tf_used))}
    
    for tf_name, data in tf_data.items():
        unix = np.array(data['unix'], dtype=np.int64)
        close = np.array(data['close'], dtype=np.float64)
        ma = np.array(data['ma'], dtype=np.float64)
        ema = np.array(data['ema'], dtype=np.float64)  
        avg_mas = np.array(data['avg_mas'], dtype=np.float64)  
        slopes = np.array(data['slopes'], dtype=np.float64)  
        

        
        
        # Normalize slope magnitudes for transparency scaling
        max_slope = np.max(np.abs(slopes))  # Get the max absolute slope
        normalized_slopes = np.abs(slopes) / max_slope  # Scale between 0 and 1
        
        # Scale alpha values from 0.1 (very transparent) to 1.0 (fully visible)
        alpha_values = 0.1 + 0.9 * normalized_slopes  
        # Insert first slope value to match length
        slopes = np.insert(slopes, 0, 0)  
        alpha_values = np.insert(alpha_values, 0, 0)  
        
        # Process and plot each timeframe
        row_positions = {tf: idx for idx, tf in enumerate(reversed(tf_used))}
        interval = config.tf_intervals[tf_name]
        relative_width = interval / config.tf_intervals[tf_used[-1]]  # Width relative to smallest timeframe
        
        for i in range(len(unix)):
            slope = slopes[i]
            alpha = alpha_values[i]  # Adjust opacity based on slope steepness
        
            # Determine color
            if slope > 0:
                color = (0, 1, 0, alpha)  # Green (RGBA)
            else:
                color = (1, 0, 0, alpha)  # Red (RGBA)
        
            # Plot bar
            base_spacing = 0.02
            bar_spacing = base_spacing * (config.tf_intervals["1-day"] / config.tf_intervals[tf_name])
            width = relative_width * (1 - bar_spacing)
            left_pos = i * relative_width + (bar_spacing * relative_width / 2)
            
            ax.barh(row_positions[tf], width, left=left_pos, height=0.8, 
                    color=color, edgecolor='black', linewidth=0.01)
        print()
    # Customize the plot
    ax.set_yticks(list(row_positions.values()))
    ax.set_yticklabels(list(row_positions.keys()))
    
    # Format x-axis to show dates
    def format_date(x, p):
        try:
            return np.datetime64(int(x), 's').astype('datetime64[m]').astype(str)
        except:
            return ''
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_date))
    #plt.xticks(rotation=45, ha='right')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    # Plot the data
    plt.figure(figsize=(25, 8), dpi=300)
    #plt.plot(unix, ma, label='200MA', color='#7A00A1', linewidth=0.3)
    #plt.plot(unix, ema, label='200EMA', color='#CFA6FF', linewidth=0.3)
    # Fill the area between the 200MA and 200EMA
    plt.fill_between(unix, ma, ema, where=ema >= ma, interpolate=True, color='lightgreen', alpha=0.15,linewidth=0.1)
    plt.fill_between(unix, ma, ema, where=ema < ma, interpolate=True, color='lightcoral', alpha=0.15,linewidth=0.1)
    plt.plot(unix, avg_mas, label='Avg (MA & EMA)', color='darkblue', linestyle='solid', linewidth=0.4)
    #plt.plot(unix, regression_line, label='Regression Line', color='blue', linewidth=0.5)
    #plt.fill_between(unix.flatten(), lower_band, upper_band, color='lightblue', alpha=0.3, label='Regression Channel')
    # Plot black dots for the closing price
    plt.plot(unix, close, '-', color='black', linewidth=0.2,alpha=0.4, label='Close Price')


    # Set the plot title and labels
    plt.title('MA Boyz')
    #plt.xlabel('Unix')
    #plt.ylabel('MAs')

    # Rotate x-axis tick labels for better readability
    plt.xticks(rotation=45)

    # Display the legend and plot the graph
    #plt.legend()
    plt.tight_layout()
    plt.yscale('log')

    
    # Adjust layout
    plt.tight_layout()
    return fig

# Usage:

fig = plot_timeframe_grid(
    tf_used=tf_used,
    start_time=start_time,
    end_time=end_time,
    candle_val=candle_val,
    config=config
)
plt.show()
   
      
"""
# Plot the data
plt.figure(figsize=(20, 10), dpi=400)
#plt.plot(unix, ma, label='200MA', color='#7A00A1', linewidth=0.3)
#plt.plot(unix, ema, label='200EMA', color='#CFA6FF', linewidth=0.3)
# Fill the area between the 200MA and 200EMA
#plt.fill_between(unix, ma, ema, where=ema >= ma, interpolate=True, color='lightgreen', alpha=0.15,linewidth=0.1)
#plt.fill_between(unix, ma, ema, where=ema < ma, interpolate=True, color='lightcoral', alpha=0.15,linewidth=0.1)
plt.plot(unix, avg_mas, label='Avg (MA & EMA)', color='darkblue', linestyle='solid', linewidth=0.4)
#plt.plot(unix, regression_line, label='Regression Line', color='blue', linewidth=0.5)
#plt.fill_between(unix.flatten(), lower_band, upper_band, color='lightblue', alpha=0.3, label='Regression Channel')
# Plot black dots for the closing price
plt.plot(unix, close, '-', color='black', linewidth=0.2,alpha=0.4, label='Close Price')


# Set the plot title and labels
plt.title('MA Boyz')
#plt.xlabel('Unix')
#plt.ylabel('MAs')

# Rotate x-axis tick labels for better readability
plt.xticks(rotation=45)

# Display the legend and plot the graph
#plt.legend()
plt.tight_layout()
plt.yscale('log')

#plt.savefig('1dMAs.png', dpi=300)
plt.show()
"""
     