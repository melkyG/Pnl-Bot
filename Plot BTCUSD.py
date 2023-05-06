import numpy as np
import h5py
import matplotlib.pyplot as plt

start_unix_time = 1604271010 
end_unix_time = 1682913060 

# Assuming the dataset is already open and loaded into memory
with h5py.File('btcpricehistorydataold - copy.hdf5', 'r+') as file:
    dataset = file['btcusd1min/btcOHLC']
    # Extract the close values from the 5th column
    #close_values = dataset[2338521:5460027, 4]
    close_values = dataset[-6000:, 4]
    #close_values[np.isnan(close_values)] = 3800
    
    # Create a time axis from the Unix timestamps in the first column
    #unix_time = dataset[10000:5460027, 0]
    unix_time = dataset[-6000:, 0]
    """    
    # Extract the close values between the start and end timestamps
    unix_time = dataset[:, 0]
    mask = (unix_time >= start_unix_time) & (unix_time < end_unix_time)
    close_values = dataset[mask, 4]
    unix_time = unix_time[mask]
    """
    time = np.array([np.datetime64(int(ts), 's') for ts in unix_time])
    
    # Plot the close values as a line chart
    plt.plot(time, close_values)
    plt.xlabel('Time')
    plt.ylabel('Close')
    plt.title('close Data')
    plt.yscale('log')
    plt.show()
