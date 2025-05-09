import h5py
import tkinter as tk
import time
import datetime
import schedule
import os
import subprocess


def unix_todate(unix_time):
    date_time = datetime.datetime.fromtimestamp(unix_time)
    return date_time.strftime('%Y-%m-%d %H:%M:%S')


def check():
    print("check() called")
    update_sync_status()
    if sync_text.get() == "Not Synced (last 5min)":
        request_data()
    root.after(5000, check)

timeframes = ["1-day/ohlc", "12-hour/ohlc", "6-hour/ohlc", "4-hour/ohlc", 
              "2-hour/ohlc", "1-hour/ohlc", "30-minute/ohlc", "15-minute/ohlc", 
              "5-minute/ohlc","3-minute/ohlc", "1-minute/ohlc"]

def update_sync_status():
    print("update_sync_status() called")
    with h5py.File('btcpricehistorydataold.hdf5', 'r+') as f:
        synced_timeframes = []
        for timeframe in timeframes:
            dset = f[timeframe]
            print
            if int(dset[-1][0]) > time.time() - 300:  # 5 minutes
                synced_timeframes.append(timeframe)

        if synced_timeframes:
            latest_synced_timeframe = synced_timeframes[0]
            latest_label.config(text="Latest Data: " + unix_todate(int(f[latest_synced_timeframe][-1][0])))
            sync_text.set("Synced (last 5min)")
        else:
            sync_text.set("Not Synced (last 5min)")
            
def request_data():
    print("request_data() called")
    if os.path.exists("sync.lock"):
        print("runnin")
        sync_text.set("Syncing...")
    else:
        open("sync.lock", "w").close()
        sync_text.set("Syncing...")
        root.update()
        subprocess.run(["python", "request data tf.py"])
        while os.path.exists("request_data.lock"):
            time.sleep(1)
        os.remove("sync.lock")
        update_sync_status()
            
root = tk.Tk()
root.title("PnL Bot 1.0")

with h5py.File('btcpricehistorydataold.hdf5', 'r+') as f:
    dset = f["1-minute/ohlc"]
    earliest_data = unix_todate(int(dset[0][0]))
    print (earliest_data)
    latest_data = unix_todate(int(dset[-1][0]))
    print (latest_data)
    
earliest_label = tk.Label(root, text="Earliest Data: " + earliest_data)
latest_label = tk.Label(root, text="Latest Data: " + latest_data)
# Set the positions of the labels using the grid layout
earliest_label.grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
latest_label.grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
# Create a label for the sync status
status_label = tk.Label(root, text="Sync Status:")
status_label.grid(row=2, column=0, padx=10, pady=5, sticky=tk.W)
# Create a text variable for the sync status text
sync_text = tk.StringVar()
# Create a label to display the sync status text
text_label = tk.Label(root, textvariable=sync_text)
text_label.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)
check()
# Run the main loop to display the window

root.mainloop()
