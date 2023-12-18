
import time
from threading import Thread, Lock
import mujoco
import mujoco.viewer
import numpy as np
import roboticstoolbox as rtb
import time
import spatialmath as sm
import numpy as np
import matplotlib.pyplot as plt
from spatialmath import UnitQuaternion
from spatialmath.base import q2r, r2x, rotx, roty, rotz

class Test:

  def __init__(self):
    self.m = mujoco.MjModel.from_xml_path('Ur5_robot/Robot_scene.xml')
    self.d = mujoco.MjData(self.m)
    self.joints = [0,0,0,0,0,0,0]
    self.dt = 1/100

    # Universal Robot UR5e kiematics parameters
    tool_matrix = sm.SE3.Trans(0., 0., 0.18)
    robot_base = sm.SE3.Trans(0,0,0)

    self.q0=[0 , -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2,0]

    self.robot = rtb.DHRobot(
        [ 
            rtb.RevoluteDH(d=0.1625, alpha = np.pi/2),
            rtb.RevoluteDH(a=-0.425),
            rtb.RevoluteDH(a=-0.3922),
            rtb.RevoluteDH(d=0.1333, alpha=np.pi/2),
            rtb.RevoluteDH(d=0.0997, alpha=-np.pi/2),
            rtb.RevoluteDH(d=0.0996)
        ], name="UR5e",
        base=robot_base,
        tool = tool_matrix,
        )
    
  #def getState(self):
    ## State of the simulater robot 
   # qState=[]    
    #for i in range(0, 6):
    #  qState.append(float(self.d.joint(f"joint{i+1}").qpos))
    #return qState

  def getObjState(self, name):
    ## State of the simulater robot    
    Objstate = self.d.body(name).xpos
    return Objstate
  
    
  def launch_mujoco(self):
    with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
      # Close the viewer automatically after 30 wall-seconds.
      start = time.time()
      while viewer.is_running():
        step_start = time.time()

        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        # with self.jointLock:
        mujoco.mj_step(self.m, self.d)

        with self.jointLock:
          if self.sendPositions:
            for i in range(0, 6):
              self.d.actuator(f"actuator{i+1}").ctrl = self.joints[i]
            self.sendPositions = False
        
        with self.GripperLock:
          if self.moveGripper:
            self.d.actuator(f"actuator{7}").ctrl = self.GrSt
            self.moveGripper = False 
    
        viewer.sync()

        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
          time.sleep(time_until_next_step)
  
  def sendJoint(self,join_values):
    with self.jointLock:
      for i in range(0, 6):
        self.joints[i] = join_values[i]
      self.sendPositions = True
    
  def controlGr(self, Gstate):
    ## 0.3 gripper open, -0.3 grpper closed, the gripper is torque controlled  
    with self.GripperLock:
      self.GrSt=Gstate
      self.moveGripper= True
      
  
  def CmoveFor(self, moveD, time):
    # moves the robot linearly in Carthesian space, robot orientaiton is fixed 
    tc = np.round(time/self.dt)
    tt= np.linspace(0,time, np.int0(tc))
    
    Transform = sm.SE3.Trans(moveD)
    trn= Transform * self.robot.fkine(self.curST)
    Trj = rtb.ctraj(self.robot.fkine(self.curST), trn, tt)
    qik = self.robot.ikine_LM(Trj, q0=self.curST)
    return qik
  
  def Cmove(self, Tr, TR, time):
    # moves the robot to pose in Carthesian space
    # R = robot transformation frame
    # TR = [1x3] translation vector 
    # time specifies the duration fo the movement 
    tc = np.round(time/self.dt)
    tt= np.linspace(0,time, np.int0(tc))
    
    Transform = sm.SE3.Rt(Tr.R,TR)
    Trj = rtb.ctraj(self.robot.fkine(self.curST), Transform, tt)
    qik = self.robot.ikine_LM(Trj, q0=self.curST)
    return qik
  
  def Jmove(self, qd, time):
    # moves the robot to a configuration in Joint space
    # qd= desired joint configuration
    # time specifies the duration fo the movement 
    tc = np.round(time/self.dt)
    tt= np.linspace(0,time, np.int0(tc))
    qik = rtb.jtraj(self.curST, qd, tt)
    return qik
  
  
  def send2sim(self, trj):
    # send trajectory step by step to simulation
    for i in trj.q:
      self.sendJoint(i)
      time.sleep(self.dt)
    self.curST = trj.q[len(trj.q)-1]  
    return self.curST  
  
  def start(self):
    self.jointLock = Lock()
    self.GripperLock = Lock()
    self.sendPositions = False
    self.moveGripper = False
    mujoco_thrd = Thread(target=self.launch_mujoco, daemon=True)
    mujoco_thrd.start()
    
    # send Robot to init congiguration
    self.sendJoint(self.q0)
    self.curST = self.q0
    input('Press to start')
    
    # gripper open
    self.controlGr(0.3)
    
    # move to object pose 
    
    Tt = self.robot.fkine(self.curST)
    print(Tt)
    Obj_pose= [0.4,0.15,0]
    
    qtrj = self.Cmove(Tt, Obj_pose, 1)
    self.send2sim(qtrj)
    
    # gripper open
    self.controlGr(-0.3)
    
    
    qtrj = self.CmoveFor([0, 0, 0.2],2)
    self.curST= self.send2sim(qtrj)
    
    self.controlGr(-0.3)
    
    Place_pose= [0.3,0.4,0.2]
    
    qtrj = self.Cmove(Tt, Place_pose, 2)
    self.curST= self.send2sim(qtrj)
    
    qtrj = self.CmoveFor([0, 0, -0.2],2)
    self.curST= self.send2sim(qtrj)
    self.controlGr(0.3)
    
    qtrj = self.CmoveFor([0, 0, 0.2],0.5)
    self.curST= self.send2sim(qtrj)
    
    qtrj = self.Jmove(self.q0, 1)
    self.curST= self.send2sim(qtrj)

   
    print("Press any key to exit")
    input()
    

if __name__ == "__main__":
  Test().start() 
  