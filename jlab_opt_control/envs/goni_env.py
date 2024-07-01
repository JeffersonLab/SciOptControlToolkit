import gymnasium as gym
from gymnasium import spaces
import numpy as np


class PolarizedBeamEnv(gym.Env):
    def __init__(self):

        self.Ebeam = 11600 #MeV

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(2,), dtype=np.float32)

        low_bounds = np.array([0, 0, 1, 1, 0, 4000, 4000, -1.5, -1.5])
        high_bounds = np.array([np.pi, np.pi, 2, 4, 45, self.Ebeam, self.Ebeam, 1.5, 1.5])

        self.observation_space = spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)
        
        # self.observation_space = spaces.Dict(
        #     {
        #         "goni_ang": spaces.Box(low=0, high=np.pi, shape=(2,)), #goni angles
        #         "plane": spaces.Discrete(2, start=1), # Orientation of the diamond radiator
        #         "mode": spaces.Discrete(4, start=1), # Orientation of the diamond radiator
        #         "phi022": spaces.Discrete(2), # Orientation of the diamond radiator [0/45]
        #         "edge": spaces.Box(low=4000, high=self.Ebeam, shape=(1,)),
        #         "req_edge": spaces.Box(low=4000, high=self.Ebeam, shape=(1,)),
        #         "beam_pos": spaces.Box(low=-1.5, high=1.5, shape=(2,)),
        #     }
        # )

        self.nsteps = 0
        self._max_episode_steps = 10 # Should be able to calculate this (not a problem as we can get the proper solution in a single step)

        self.states, _ = self.reset()
        

    def _get_obs(self):
        return np.array([self.pitch, self.yaw, self.plane, self.mode, self.phi022, self.edge, self.req_edge, self.beam_pos_x, self.beam_pos_y]) # beam position is two variables
        # return {"goni_ang": [self.pitch, self.yaw], "plane": self.plane, "mode": self.mode, "phi022": self.phi022, "edge": self.edge, "req_edge": self.req_edge, "beam_pos": self.beam_pos}

        
    def reset(self, seed=None):
        super().reset(seed=seed)

        # Utilize historical data to set parameters on the reset (within the dataset)
        
        # 99% of these values would be useless using random selection between 0/pi
        # Set range for pitch and yaw around value of c angle within bounds
        # Also a function of phi022
        self.pitch = self.np_random.uniform(0, np.pi)
        self.yaw = self.np_random.uniform(0, np.pi)

        self.plane = self.np_random.integers(1,2)
        self.mode = self.np_random.integers(1,4)
        self.phi022 = self.np_random.integers(1,2) # Planar polarization (direction of one of the vectors) 
        self.edge = self.np_random.uniform(7000, 9500)
        self.req_edge = 8420 # If you know this value then you can calculate the c angle value
        
        self.beam_pos_x = -0.5
        self.beam_pos_y = 1.0
        
        observation = self._get_obs()

        self.nsteps = 0
        
        return observation, {}


    def new_edge(self, delta_c):
        delta_c = np.deg2rad(delta_c)

        k = 26.5601 #MeV
        g = 2
        E0 = self.Ebeam
        Ei = self.edge

        
        Ef = E0*(1-1/((delta_c*g*E0)/k + 1/(1-Ei/E0)))

        return Ef


    def get_delta_c_from_delta_pitch_yaw(self, pitch, yaw):
        if self.plane==1:
            if self.mode==2 or self.mode==3:
                if pitch>=0 and yaw>=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw<=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)
                else:
                    c=0
            else:
                if pitch>=0 and yaw>=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw<=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)
                else:
                    c=0
        else:
            if self.mode==1 or self.mode==4:
                if pitch>=0 and yaw<=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw>=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)
                else:
                    c=0
            else:
                if pitch>=0 and yaw<=0:
                    c = -np.sqrt(pitch*pitch + yaw*yaw)
                elif pitch<=0 and yaw>=0:
                    c = np.sqrt(pitch*pitch + yaw*yaw)
                else:
                    c=0
        return c
    
    def get_delta_c_from_delta_E(self):
        k = 26.5601 #MeV
        g = 2
        E0 = self.Ebeam
        Ei = self.edge
        Ef = self.req_edge
        
        delta_c = (k/g)*(Ef-Ei)/((E0-Ei)*(E0-Ef)) #in radians
        
        return delta_c

    
    def get_optimal_action(self):
        #from moveCbrem.sh script
        c=self.get_delta_c_from_delta_E() #radians

              
        phi=self.phi022*np.pi/4
        cosphi=np.cos(phi)
        sinphi=np.sin(phi)

        if self.plane==1:
            if self.mode==2 or self.mode==3:
                v = + c*cosphi
                h = + c*sinphi
            else:
                v = - c*cosphi
                h = - c*sinphi
        else:
            if self.mode==1 or self.mode==4:
                v = + c*sinphi
                h = - c*cosphi
            else:
                v = - c*sinphi
                h = + c*cosphi
                
        pitch_change = np.rad2deg(h)
        yaw_change = np.rad2deg(v)

        
        return np.array([pitch_change, yaw_change])

    
    def step(self, action):
        
        self.nsteps += 1

        self.pitch += action[0]
        self.yaw += action[1]        

        new_edge = self.new_edge(self.get_delta_c_from_delta_pitch_yaw(action[0], action[1]))

        reward = np.absolute(self.edge - self.req_edge) - np.absolute(new_edge - self.req_edge) # Change in delta E
        # Reward is a function of your current step and your previous step
        # reward = 1./(np.absolute(new_edge - self.req_edge)+0.00001)        

        self.edge = new_edge

        # Episode is done if the edge is within some window of the required edge
        if np.absolute(self.edge-self.req_edge)/self.req_edge < 0.0001:
            done=True
        else:
            done=False

        if self.nsteps >= self._max_episode_steps:
            done=True

        self.state = self._get_obs()
            
        return self.state, reward, done, False, {} # Look into truncation
        

        
    def render(self):
        pass

