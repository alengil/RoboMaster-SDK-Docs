from robomaster import robot
import time

def main():
    ep_robot = robot.Robot()
    ep_robot.initialize(conn_type="ap")
    
    # בדיקת תנועה
    ep_robot.chassis.move(x=-5, y=0, z=0)
    time.sleep(2)
    
    ep_robot.close()

if __name__ == '__main__':
    main()
