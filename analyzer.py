from iterator import iterate
import config
from open_hdf5 import open_hdf5, close_hdf5
import random
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.linear_model import LinearRegression
from functions import smooth_series, determine_bias
from scipy.signal import savgol_filter 


#select timeframes
tf_used = config.all_timeframes[-8:]

with open_hdf5(config.btcusd, 'r') as f:
    dset = f[tf_used[0]+ "/ohlc"]
    start = dset[0, 0]
    end = dset[-1, 0]

for tf in tf_used:
    unix1, close = iterate(start, end, tf, 'ohlc')
    unix2, ma = iterate(start, end, tf, '200ma')
    unix3, ema = iterate(start, end, tf, '200ema')
    print("-"*5, tf, "-"*5)
    print("first: ", unix1[0], close[0], ma[0], ema[0])
    print("last: ", unix1[-1], close[-1], ma[-1], ema[-1])
    
# trend_rider.py



def main():
    # Initialize

    # Get the current market state
    outlooks = 2

    # Output the current market state
    print("Current Market State:")
    #print(f"Low Timeframe Trend: {outlooks['low_timeframe_trend']}")
    #print(f"High Timeframe Trend: {outlooks['high_timeframe_trend']}")
    #print(f"Very High Timeframe Trend: {outlooks['very_high_timeframe_trend']}")
    print("\nDetailed Trends:")
    #print(f"Low Timeframe Data: {outlooks['low_timeframe_data']}")
    #print(f"High Timeframe Data: {outlooks['high_timeframe_data']}")
    #print(f"Very High Timeframe Data: {outlooks['very_high_timeframe_data']}")

if __name__ == "__main__":
    main()


class Price:
    def __init__(self, tf: str, unix: int, bias: float, stretch: float):
        self.tf = tf
        self.unix = unix
        self.bias = bias
        self.stretch = stretch

    def __repr__(self):    
        return
    
class Trend:
    def __init__(self, birth_unix, initial_price):
        self.birth_unix = birth_unix
        self.death_unix = None
        self.magnitude = 0
        self.direction = 0
        self.actionable = False
        self.prices = [initial_price]
        
    def __repr__(self):    
        return



candle_val = 4

#input range
scan_range = (1514595600+0*(0.5*86400),1741564800-2*(0.5*86400))#(1322395200-2*(0.5*86400),1739404800+(0*86400))#(1684800000+(47*60*60*4), 1684800000+(55*60*60*4)) #(1734537600, 1735228800)#(1325318400, 1325318400 + (100*60*60*4)) #(1684800000, 1684800000+(86400*50)) 
tf_data = {}    
debug_rows = 5

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

        # Access the datasets
        
        unix_stamps = f['1-day/ohlc'][:, 0]
        close_prices = f['1-day/ohlc'][:, 4]
        low_prices = f['1-day/ohlc'][:, 3]
        high_prices = f['1-day/ohlc'][:, 2]
        ma_values = f['1-day/200ma'][:, 1]
        ema_values = f['1-day/200ema'][:, 1]
        
        # Define the Unix range
        unix_range_start = (1436054400 + 84000)
        unix_range_end = 1641373200
        
        # Filter rows within the provided Unix range
        valid_rows = (unix_stamps >= unix_range_start) & (unix_stamps <= unix_range_end)
        filtered_rows = valid_rows.nonzero()[0]
        
        # Filtered arrays
        filtered_unix_stamps = unix_stamps[filtered_rows]
        filtered_close_prices = close_prices[filtered_rows]
        filtered_low_prices = low_prices[filtered_rows]
        filtered_high_prices = high_prices[filtered_rows]
        filtered_ma_values = ma_values[filtered_rows]
        filtered_ema_values = ema_values[filtered_rows]
        #print(filtered_unix_stamps)
        #print(len(filtered_unix_stamps))
        #print (filtered_close_prices[0], filtered_ma_values[0], filtered_ema_values[0])
        #for i in range(len(filtered_rows)):
            #print(i)
        # Create lists to store filtered data points
        list_unix = []
        list_close = []
        list_ma = []
        list_ema = []
    
        # Iterate through the filtered arrays and filter further
        for i in range(len(filtered_rows)):
            if (
                filtered_close_prices[i] == max(filtered_close_prices[i], filtered_ma_values[i], filtered_ema_values[i])
                or filtered_close_prices[i] == min(filtered_close_prices[i], filtered_ma_values[i], filtered_ema_values[i])
            ):
                list_unix.append(filtered_unix_stamps[i])
                list_close.append(filtered_close_prices[i])
                list_ma.append(filtered_ma_values[i])
                list_ema.append(filtered_ema_values[i])
        #print (list_unix, list_close, list_ma, list_ema)
        # Size 2 window iteration
        window_size = 2
        trends = []
        birth_date = None
        birth_price = None
    
        # Iterate through the filtered lists
        for i in range(len(list_unix) - window_size + 1):
            # Get the window data
            #print(i)
            window_unix = list_unix[i : i+window_size]
            window_close = list_close[i : i+window_size]
            window_ma = list_ma[i : i+window_size]
            window_ema = list_ema[i : i+window_size]
            #print (window_unix[0], window_close[0], window_ma[0], window_ema[0])
            # Check if it's an uptrend start
            if (
                window_close[1] > max(window_ma[1], window_ema[1])
                and window_close[0] < min(window_ma[0], window_ema[0])
                ):
                print(window_close[1])
                birth_date = int(window_unix[1])
                birth_price = max(window_ma[1], window_ema[1])
                # Initialize trend end variables
                death_date = None
                death_price = None
                
            # Check if it's an uptrend end
            if (
                birth_date is not None
                and birth_price is not None
                and window_close[0] > max(window_ma[0], window_ema[0])
                and window_close[1] < min(window_ma[1], window_ema[1])
            ):
                print(window_close[1])
                death_date = int(window_unix[0])
                death_price = window_close[0] #min(window_ma[1], window_ema[1])
                print (death_date, death_price)
                # Check if both start and end points are found and end price is 3% or more greater than start price
                if death_date is not None and death_price is not None and (death_price - birth_price) / birth_price >= 0.007:
                    filtered_rows = (filtered_unix_stamps >= birth_date) & (filtered_unix_stamps <= death_date)
    
                    # Extract the relevant data within the specified range
                    relevant_high_prices = filtered_high_prices[filtered_rows]
                    
                    # Find the highest price within the range
                    highest_price = max(relevant_high_prices)
                    timeframe = '1-day'  # Modify this accordingly
                    direction = 'Up'  # Modify this accordingly
                    magnitude = ((death_price/birth_price)-1)*100
                                #within unix of trend, trend_high/trend_low
                    trend = Trend(birth_date, death_date, birth_price, death_price, timeframe, direction, magnitude)
                    trends.append(trend)
    
        # Print the trends
        for trend in trends:
            print(f"Timeframe: {trend.timeframe}"" ||", f"Direction: {trend.direction}"" ||", f"Magnitude: {trend.magnitude:.1f}%")
            print(f"Start Unix: {trend.birth_date}, Start Price: {trend.birth_price:.0f}")
            print(f"End Unix: {trend.death_date}, End Price: {trend.death_price:.0f}")
            print()
"""