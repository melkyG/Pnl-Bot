import random
import numpy as np
import config
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sklearn.linear_model import LinearRegression
from functions import smooth_series, determine_bias
from scipy.signal import savgol_filter 

#select timeframes
tf_used = config.all_timeframes[-1:]
candle_val = 4

#input range
scan_range = (1514595600+0*(0.5*86400),1741564800-2*(0.5*86400))#(1322395200-2*(0.5*86400),1739404800+(0*86400))#(1684800000+(47*60*60*4), 1684800000+(55*60*60*4)) #(1734537600, 1735228800)#(1325318400, 1325318400 + (100*60*60*4)) #(1684800000, 1684800000+(86400*50)) 
tf_data = {}    
debug_rows = 10

# Define which smoothing method to use
smoothing_method = "SG"  # Options: "SMA", "EMA", "SG" (Savitzky-Golay)
slope_smoothing_window = 5  # Adjust based on your needs
ema_alpha = 0.05 # Alpha value for EMA smoothing (only used if smoothing_method = "EMA")

# Initialize lists to store slope values for smoothing
slope_values = []
smoothed_slope_values = []
# Define which plots to include
plot_flags = {
    "price_chart": True,  # Toggle for price chart (log-scale)
    "slope_oscillator": False,  # Toggle for slope indicator
    "slope_rate": True  # Toggle for slope rate (acceleration)
}
slope_intensity = False #velo plot
smooth_slope_rate = True #accel plot
with h5py.File('btcpricehistorydataold.hdf5', 'r') as f:
    for tf in tf_used:
        adjusted_debug_rows = int(debug_rows*config.tf_intervals["1-day"]/config.tf_intervals[tf])
        print("adjusted debug rows: ",adjusted_debug_rows)
        start_time, end_time = scan_range
        print(f"user inputs: ({start_time}, {end_time}) length: ", end_time-start_time, type(start_time), type(end_time),"\n")

        #adjust input range if outside of dataset
        if scan_range[0] < f[f"{tf}/ohlc"][0][0] : start_time = f[f"{tf}/ohlc"][0][0] 
        if scan_range[1] > f[f"{tf}/ohlc"][-1][0] : end_time = f[f"{tf}/ohlc"][-1][0]
        print(f"Database Boundaries Applied: ({start_time}, {end_time}) length: ", end_time-start_time, type(start_time), type(end_time),"\n")
        print("="*22,tf,"="*22)
        interval = config.tf_intervals[tf]  # Get the interval for the current timeframe
        start_time = int(((start_time + (interval - 1)) // interval) * interval)+interval
        end_time = (((end_time + (interval - 1)) // interval) * interval)#1739486880
        print(f"adj. unix range for processing ({start_time}, {end_time}) length: ", end_time-start_time,"\n")
        # Load datasets using time filtering
        ohlc_data = f[f"{tf}/ohlc"]
        ma_data = f[f"{tf}/200ma"]
        ema_data = f[f"{tf}/200ema"]
        
        # Find indices where UNIX timestamps fall within the range
        start_idx = (ohlc_data[:, 0] >= start_time).argmax()  # First index that meets the condition
        end_idx = (ohlc_data[:, 0] <= end_time).argmin()      # Last index that meets the condition
        
        # Slice the data using computed indices
        ohlc_data = ohlc_data[start_idx:end_idx + 1]
        ma_data = ma_data[start_idx:end_idx + 1]
        ema_data = ema_data[start_idx:end_idx + 1]
        
        smoothed_ma = smooth_series(ma_data[:, 1], method=smoothing_method, window=slope_smoothing_window, ema_alpha=ema_alpha)
        smoothed_ema = smooth_series(ema_data[:, 1], method=smoothing_method, window=slope_smoothing_window, ema_alpha=ema_alpha)

        # Initialize the list for this timeframe
        tf_data[tf] = []
        
        #Collect raw data       
        for i in range(len(ohlc_data)):
            unix, close, ma, ema = ohlc_data[i, 0], ohlc_data[i, 4], smoothed_ma[i], smoothed_ema[i]
            avg_mas = (ma + ema) / 2
            bias = determine_bias(i, ohlc_data, ma_data, ema_data)
            stretch = 100 * (close / avg_mas - 1)       
            tf_data[tf].append({
                "unix": unix,
                "close": close,
                "ma": ma,
                "ema": ema,
                "avg_mas": avg_mas,
                "bias": bias,
                "stretch": stretch,
            })
        
        #Compute Slope 
        slope_values = [0]  # Start with 0 for the first item
        for i in range(len(tf_data[tf])):
            if i > 0:
                prev_avg_mas = tf_data[tf][i - 1]["avg_mas"]
                prev_unix = tf_data[tf][i - 1]["unix"]
                avg_mas = tf_data[tf][i]["avg_mas"]
                unix = tf_data[tf][i]["unix"]
                
                delta_time = unix - prev_unix
                delta_mas = avg_mas - prev_avg_mas
                
                # Compute percentage slope (preventing division by zero)
                percentage_slope = (delta_mas / prev_avg_mas) * 100 if prev_avg_mas != 0 else 0
            else:
                percentage_slope = 0  # First row has no previous value to compare
        
            slope_values.append(percentage_slope)
            tf_data[tf][i]["percentage_slope"] = slope_values[i] 
        #Compute Slope Rate
        for i in range(len(tf_data[tf])):
            if i > 0:
                prev_slope = slope_values[i - 1]
                delta_time = tf_data[tf][i]["unix"] - tf_data[tf][i - 1]["unix"]
                slope_rate = (slope_values[i] - prev_slope) / delta_time if delta_time != 0 else 0
            else:
                slope_rate = 0  # First row has no previous slope to compare

            # Append computed values to existing dictionary
                       
            tf_data[tf][i]["slope_rate"] = slope_rate
            # Extract slope_rate values
        slope_rates = np.array([entry["slope_rate"] for entry in tf_data[tf]])
        
        if smooth_slope_rate == True:
            smoothed_slope_rates = smooth_series(slope_rates, method="SMA", window=5, ema_alpha=0.2)
            # Assign smoothed values back to tf_data
            for i, entry in enumerate(tf_data[tf]):
                entry["slope_rate"] = smoothed_slope_rates[i]
        
    # Extract data from tf_data
    unix_values = [entry["unix"] for entry in tf_data[tf]]
    close_values = [entry["close"] for entry in tf_data[tf]]
    slope_values = [entry["percentage_slope"] for entry in tf_data[tf]]
    slope_rate_values = [entry["slope_rate"] for entry in tf_data[tf]]  # Extract slope_rate
    avg_ma_values = [entry["avg_mas"] for entry in tf_data[tf]]
    


for tf in tf_used:
    print(f"**{tf}**")
    
    # Assuming all dictionaries have the same keys, get keys from the first item
    if tf_data[f"{tf}"] and isinstance(tf_data[f"{tf}"][-1], dict):
        headers = list(tf_data[f"{tf}"][-1].keys())
       
        for header in headers:
            print(f"{header:<15}", end="")
        print("\n" + "-" * (8 + 15 * len(headers)))
        
        for i, value in enumerate(tf_data[f"{tf}"][-adjusted_debug_rows:]):

            for header in headers:
                val = value.get(header, '')
                # Format specific float values to 2 decimal places
                if header in ["ma", "ema", "avg_mas", "stretch", "percentage_slope","slope_rate"] and isinstance(val, float):
                    formatted_val = f"{val:.2f}"
                    print(f"{formatted_val:<15}", end="")
                else:
                    print(f"{val:<15}", end="")
            print()
        print()
        
        
  
"""        
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
"""