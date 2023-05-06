import requests
import time
import csv
import h5py
import datetime
import numpy as np
import os

# Create a lock file
lock_file = "request_data.lock"
with open(lock_file, "w") as f:
    f.write("lock")

api_key = 'QqcWuai88z3kdeVI3UD1a9JLShTzhqpW'

def unix_todate(unix_time):
    date_time = datetime.datetime.fromtimestamp(unix_time)
    return date_time.strftime('%Y-%m-%d %H:%M:%S')


with h5py.File('btcpricehistorydataold - Copy.hdf5', 'r+') as f:
    dset = f["btcusd1min/btcOHLC"]
    earliest_unix = int(dset[0][0])
    latest_unix = int(dset[-1][0])
    print("Latest Date: ", latest_unix, (unix_todate(latest_unix)))
    
start_time = latest_unix #1420449180 #142037440
end_time = int(time.time())   #latest_unix + 60
step = 60

#number of minutes in unix time range
num_mins = int((end_time-start_time)/60)

print('unix time range:', num_mins, 'minutes', ',', int(num_mins/60), 'hours', ',', int(num_mins/1440), 'days')

if num_mins >= 1000:
    limit = 1000
else:
    limit = int(num_mins)    
print('limit parameter =',limit)
    
url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"

# Create a list to store the data
data = []
start = start_time 
end = start_time +  limit*step 

# Calculate the number of requests needed
num_requests = 1 if num_mins <= 1000 else ((num_mins - 1) // 1000 + 1)
print('requests needed: ',num_requests)
print("============================")

for i in range(num_requests):
        if num_mins >= 1000:
            limit = 1000
        else:
            limit = int(num_mins)  
        if i == num_requests - 1:
            limit = num_mins-(i*limit)
        print('batch', i+1, 'limit = ', limit)
        request_url = f"{url}?step={step}&limit={limit}&start={start}&end={end}&apikey={api_key}"
        response = requests.get(request_url)
        time.sleep(1.2)
        #print(response.json())
        print(start + 60, end,", ", 'mins diff: ', int(end-start-60)/60)
        print('results returned = ' ,len(response.json()['data']['ohlc']))
        #print('first timestamp: ', response.json()['data']['ohlc'][0]['timestamp'])
        #print('last timestamp: ', response.json()['data']['ohlc'][-1]['timestamp'])
        # Append the data to the list
        data.extend(response.json()['data']['ohlc'])
        
        print("\n")
        # Calculate the start and end times for the request
        #start = start + limit*step + 60
        if len(data) > 0:
            start = int(data[-1]['timestamp'])
        else:
            start = start_time

        #end = start_time
        end = min(start + limit*step, end_time)
        
print("============================")


print('Total response length: ', len(data))
print('first timestamp: ',data[0]['timestamp'])
print('last timestamp: ',data[-1]['timestamp'])
print('mins diff: ', (int(data[-1]['timestamp'])-int(data[0]['timestamp']))/60)


with h5py.File('btcpricehistorydataold - Copy.hdf5', 'r+') as f:
    dset = f["btcusd1min/btcOHLC"]
    
    current_shape = dset.shape[0]

    # Get the data from the JSON response
    #data = response.json()['data']['ohlc']

    # Create a numpy array from the data
    data_array = np.zeros((len(data), 6))
    for i, row in enumerate(data):
        data_array[i] = [float(row['timestamp']), float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume'])]

    # Resize the dataset to make room for the new data
    dset.resize(current_shape + len(data), axis=0)

    # Append the new data to the dataset
    dset[current_shape:] = data_array

# Remove the lock file when the script is done
os.remove(lock_file)

"""
with h5py.File('btcpricehistorydataold - Copy.hdf5', 'r+') as f:
    dset = f["btcusd1min/btcOHLC"]
    earliest_unix = int(dset[0][0])
    latest_unix = int(dset[-1][0])
    print("Latest Date: ", latest_unix, (unix_todate(latest_unix)))
"""

"""
# Write the data to a CSV file
with open('btcusd_ohlc_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    for d in data:
        writer.writerow([d['timestamp'], d['open'], d['high'], d['low'], d['close'], d['volume']])
"""