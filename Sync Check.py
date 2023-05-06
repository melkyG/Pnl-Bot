import h5py
import tkinter as tk
import time
import datetime
import schedule
import os
import subprocess

indexed_unixdata = {}

def unix_todate(unix_time):
    date_time = datetime.datetime.fromtimestamp(unix_time)
    return date_time.strftime('%Y-%m-%d %H:%M:%S')

# Create a function to update the sync status text
def update_sync_status():
    with h5py.File('btcpricehistorydataold - Copy.hdf5', 'r+') as f:
        dset = f["btcusd1min/btcOHLC"]
        latest_unix = int(dset[-1][0])
        if latest_unix > time.time() - 300: #3600
            sync_text.set("Synced (last 5min).")
        else:
            sync_text.set("Not Synced (last 5min)")
       
def request_data():
    """
    if os.path.exists("sync.lock"):
        sync_text.set("Syncing...")
        #return
    # Create lock file
    open("sync.lock", "w").close()
    """
    # Display "Syncing..." in the sync_text label
    sync_text.set("yo...")
    root.update()

    # Run request_data.py using subprocess
    subprocess.run(["python", "request_data.py"])

    # Wait for request_data.py to delete the lock file
    while os.path.exists("request_data.lock"):
        time.sleep(1)
    """
    # Remove lock file
    os.remove("sync.lock")
    """
    # Update the sync status
    update_sync_status() 

# Create the main window
root = tk.Tk()
root.title("PnL Bot 1.0")

with h5py.File('btcpricehistorydataold - Copy.hdf5', 'r+') as f:
    dset = f["btcusd1min/btcOHLC"]
    earliest_unix = int(dset[0][0])
    latest_unix = int(dset[-1][0])
    earliest_data = unix_todate(earliest_unix)
    latest_data = unix_todate(latest_unix)
    
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
#sync_text.set("Not Synced")

# Create a label to display the sync status text
text_label = tk.Label(root, textvariable=sync_text)
text_label.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

# Update the sync status text
update_sync_status()


# Check if data needs to be requested
if sync_text.get() == "Not Synced (last 5min)":
    request_data()


# Run the main loop to display the window
root.mainloop()


"""
def append_data():
    with h5py.File("data.hdf5", "a") as f:
        # append data to the file here

def read_data():
    with h5py.File("data.hdf5", "r") as f:
        # read data from the file here

# Schedule the append_data function to run every Unix hour (3600 seconds)
schedule.every(3600).seconds.do(append_data)

# Run the read_data function continuously
while True:
    read_data()
    time.sleep(1)
    schedule.run_pending()


In this example, we define two functions: append_data() and read_data(). 
The append_data() function opens the HDF5 file in append mode and appends data to it, 
while the read_data() function opens the file in read-only mode and reads data from it.

We then use the schedule module to schedule the append_data() function to run every Unix hour 
(3600 seconds) using the schedule.every() method. We also use a while loop to ontinuously run 
the read_data() function, and use the time.sleep() method to add a delay of 1 second between 
each iteration of the loop. Finally, we use the schedule.run_pending() method to check if 
there are any scheduled tasks to run, and run them if there are.

This way, your program will continuously read data from the HDF5 file while also appending 
data to it every Unix hour.
"""