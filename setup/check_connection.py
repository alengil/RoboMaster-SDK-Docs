from robomaster import robot
import time
def main():
   ep_robot = robot.Robot()
   ep_robot.initialize(conn_type="ap")


   # battery = ep_robot.battery
   # val = battery.get_battery().wait_for_completed()
   # print("Battery: {0}%".format(val))
   ep_robot.chassis.move(x=-5, y=0, z=0)
   # ep_robot.battery.get_battery()
   time.sleep(2)
   ep_robot.close()


if __name__ == '__main__':
   main()
