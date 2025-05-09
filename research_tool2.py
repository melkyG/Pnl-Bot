import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import config

def find_price_velocity(price_change: tuple[float,float], scan_range: tuple[int, int], time_change: int, timeframe: str, direction: str):
    """
    Identifies a price speed within specified time range using 2-slot windows.
    Each window consists of two points exactly time_change apart.

    Parameters:
    - price_change: float - percentage change to look for (e.g., 50.0 for 50% move)
    - scan_range: tuple - (start_time, end_time) unix timestamps to scan
    - time_change: int - required time difference between slots (e.g., 86400*30 for 30 days)
    - timeframe: str - data timeframe to use
    - direction: str - 'up' or 'down' for price movement direction
    """
    file_path = 'btcpricehistorydataold.hdf5'
    dataset_path = f'{timeframe}/ohlc'
    
    # Define a small tolerance (e.g., 1 second) for exact match
    TOLERANCE = 1  
    
    try:
        with h5py.File(file_path, 'r') as f:           
            dset = f[dataset_path]
            interval = config.tf_intervals[timeframe]
            start_time = ((scan_range[0] + (interval - 1)) // interval) * interval
            end_time = ((scan_range[1] + (interval - 1)) // interval) * interval
            filtered_data = [row for row in dset if start_time <= row[0] <= end_time]
            filtered_data = np.array(filtered_data)
            print(f"Filtered data contains {len(filtered_data)} rows within unix range: {scan_range}.")

            # Analyze time differences in data
            unix_column = filtered_data[:, 0]
            diffs = np.diff(filtered_data[:, 0])
            print("Time difference statistics (in seconds):")
            print(f"Min: {np.min(diffs)}")
            print(f"Max: {np.max(diffs)}")
            print(f"Mean: {np.mean(diffs)}")
            print(f"Median: {np.median(diffs)}")
            
            filtered_rows = []
            last_window_end = None
            count = 1
            #if price_change[1] > price_change[0]:
            #    price_change_a, price_change_b = price_change[0], price_change[1]
            #print(price_change)

            # For each potential starting point
            for start_idx in range(len(filtered_data)):
                start_time = filtered_data[start_idx][0]
                target_time = start_time + time_change
                
                # Find the matching end point
                end_idx = None
                for j in range(start_idx + 1, len(filtered_data)):
                    current_diff = filtered_data[j][0] - start_time
                    if abs(current_diff - time_change) <= TOLERANCE:
                        end_idx = j
                        break
                    elif current_diff > time_change:
                        # If we've passed our target time without finding a match,
                        # no need to look further
                        break
                
                if end_idx is None:
                    continue  # No exact match found, skip this window
                
                # Create 2-slot window
                window = np.array([filtered_data[start_idx], filtered_data[end_idx]])
                current_start = window[0][0]
                current_end = window[-1][0]
                
                # Skip if this window overlaps with a previously found pattern
                #if last_window_end is not None and current_start <= last_window_end:
                #    count += 1
                #    continue
                
                
                # Check price conditions
                price1 = window[0, 1]
                price2 = window[-1, 1]
                
                if direction == 'up':
                    target_price = price1 * (1 + (price_change[0]/100))
                    target_price2 = price1 * (1 + (price_change[1]/100))
                    if price_change[0] == price_change[1]:
                        if price2 >= target_price:
                            filtered_rows.append(window[0, 0])
                            last_window_end = current_end
                            print(f"\nWindow #{count}")
                            print(f"Start: unix: {window[0][0]}, Open: {window[0][1]}")
                            print(f"End: unix: {window[-1][0]}, Open: {window[-1][1]}")
                            print(f"Time difference: {window[-1][0] - window[0][0]}")
                            print("Price Difference: ",(window[-1][1]/window[0][1]))
                    else:
                        if price2 >= target_price and price2 <= target_price2:
                            filtered_rows.append(window[0, 0])
                            last_window_end = current_end
                            print(f"\nWindow #{count}")
                            print(f"Start: unix: {window[0][0]}, Open: {window[0][1]}")
                            print(f"End: unix: {window[-1][0]}, Open: {window[-1][1]}")
                            print(f"Time difference: {window[-1][0] - window[0][0]}")
                            print("Price Difference: ",(window[-1][1]/window[0][1]))
                        
                elif direction == 'down':
                    target_price = price1 * (1 - (price_change[0]/100))
                    target_price2 = price1 * (1 - (price_change[1]/100))
                    if price_change[0] == price_change[1]:
                        if price2 <= target_price:
                            filtered_rows.append(window[0, 0])
                            last_window_end = current_end
                            print(f"\nWindow #{count}")
                            print(f"Start: unix: {window[0][0]}, Open: {window[0][1]}")
                            print(f"End: unix: {window[-1][0]}, Open: {window[-1][1]}")
                            print(f"Time difference: {window[-1][0] - window[0][0]}")
                            print("Price Difference: ",(window[-1][1]/window[0][1]))
                    else:
                        if price2 <= target_price and price2 >= target_price2:
                            filtered_rows.append(window[0, 0])
                            last_window_end = current_end
                            print(f"\nWindow #{count}")
                            print(f"Start: unix: {window[0][0]}, Open: {window[0][1]}")
                            print(f"End: unix: {window[-1][0]}, Open: {window[-1][1]}")
                            print(f"Time difference: {window[-1][0] - window[0][0]}")
                            print("Price Difference: ",(window[-1][1]/window[0][1]))
                
                count += 1
            
            print(f"\nStarting at Unix timestamp(s):")
            for timestamp in filtered_rows:
                print(f"{int(timestamp)}")
            if price_change[0] == price_change[1]:
                print(price_change[0], price_change[1])
                print(f"\nFound {len(filtered_rows)} instances of {price_change[0]}% (or greater) {direction}ward price movement over {time_change} seconds")
            else:
                print(f"\nFound {len(filtered_rows)} instances of {price_change[0]}-{price_change[1]}% {direction}ward price movement over {time_change} seconds")
            return filtered_rows
            
    except Exception as e:
        print(f"Couldn't Access HDF File: {e}")
    pass