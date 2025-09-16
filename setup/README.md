# Robomaster SDK Setup Guide

This guide provides step-by-step instructions for setting up the Robomaster SDK development environment.

## Prerequisites

- Ubuntu/Debian-based Linux system
- Terminal access with sudo privileges
- VSCode (recommended)

## Setup Instructions

### 1. Install Python 3.8

Run the following commands in your terminal to install Python 3.8:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-distutils
```

### 2. Robomaster SDK Installation

Open terminal in VSCode and execute the following commands:

1. **Create virtual environment:**
   ```bash
   python3.8 -m venv ~/robomaster-env
   ```

2. **Activate virtual environment:**
   ```bash
   source ~/robomaster-env/bin/activate
   ```

3. **Upgrade pip and install dependencies:**
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

4. **Install Robomaster SDK:**
   ```bash
   pip install robomaster
   ```

## Connection Test

Use the following test script to verify your Robomaster connection, Also at [check_connection.py](setup/check_connection.py):

```python
from robomaster import robot
import time

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")

    # Optional: Check battery level
    # battery = ep_robot.battery
    # val = battery.get_battery().wait_for_completed()
    # print("Battery: {0}%".format(val))
    
    # Test movement
    ep_robot.chassis.move(x=-5, y=0, z=0)
    time.sleep(2)
    ep_robot.close()

if __name__ == '__main__':
    main()
```

### Test Script Explanation

- **Initialize robot:** Connects to Robomaster via AP mode
- **Movement test:** Moves robot 5 units backward on x-axis
- **Battery check:** (Commented) Optional battery level monitoring
- **Cleanup:** Properly closes robot connection

## Usage Notes

### Virtual Environment Management

- **Activate environment:** `source ~/robomaster-env/bin/activate`
- **Deactivate environment:** `deactivate`
- Always activate the virtual environment before running Robomaster scripts

### Connection Types

- **AP Mode:** `conn_type="ap"` - Direct WiFi connection to robot
- **STA Mode:** `conn_type="sta"` - Connect via router (alternative option)

### Troubleshooting

1. **Permission errors:** Ensure you have sudo privileges
2. **Connection issues:** Verify robot WiFi connection and IP address
3. **Import errors:** Confirm virtual environment is activated
4. **Movement not working:** Check robot battery level and surface conditions
