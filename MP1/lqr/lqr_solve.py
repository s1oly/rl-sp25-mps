import numpy as np

def get_system(env):
    if env.unwrapped.spec.id == 'PendulumInvert-v1':
        system = PendulumBalance(dt=0.05)
    if env.unwrapped.spec.id == 'PendulumBalance-v1':
        system = PendulumBalance(dt=0.05)
    if env.unwrapped.spec.id == 'DoubleIntegrator-v1':
        system = DoubleIntegrator(dt=0.05)
    return system

class LQRControl(object):
    def __init__(self, env, state=None):
        self.system = get_system(env)
        A, B, Q, R = self.system.get_system()
        self.lqr = LQRSolver(A, B, Q, R, 200)
        self.lqr.solve()
        self.step = 0
        self.action_space = env.action_space

    def act(self, state):
        u = self.action_space.sample()
        # TODO: Once you finish implementing LQRSolver, uncomment the following
        # line to use it to control the system 
        u = self.lqr.get_control(state, self.step)
        self.step += 1
        return u

class LQRSolver(object):
    def __init__(self, A, B, Q, R, T):
        self.A, self.B, self.Q, self.R = A, B, Q, R
        self.T = T

    def solve(self):
        # TODO: Implement the solver here, i.e. compute the time-varying K_i and P_i matrices
        P_i = {}
        P_i[self.T] = self.Q
        K_i = {}
        for i in range (1, self.T + 1):
            K_i[self.T - i] = np.linalg.inv(self.R + self.B.T @ P_i[self.T - i + 1] @ self.B) @ self.B.T @ P_i[self.T - i + 1] @ self.A
            P_i[self.T - i] = self.Q + self.A.T @ P_i[self.T - i + 1] @ self.A - self.A.T @ P_i[self.T - i + 1] @ self.B @ np.linalg.inv(self.R +self.B.T @ P_i[self.T - i + 1] @ self.B) @ self.B.T @ P_i[self.T - i + 1] @ self.A
        self.K_i = K_i
        self.P_i = P_i

    def get_control(self, x, i):
        # TODO: Implement code for mapping states to actions using the
        # pre-computed K_i and P_i matrices 
        control = -1 * self.K_i[i] @ x
        return control

class DoubleIntegrator(object):
    def __init__(self, dt):
        self.dt = dt
        None
    
    def get_system(self):
        # TODO: Return A, B, Q, R for this system
        A = np.array([[1, self.dt], [0, 1]])
        B = np.array([[0], [self.dt]])
        # Need some Q and R to get sum of 2 diagnol matrix to obtain x^2 + v^2 + u^2
        Q = np.array([[1,0],[0,1]])
        R = np.array([[1]])
        return A, B, Q, R

class PendulumBalance(object):
    def __init__(self, dt):
        self.dt = dt
    
    def get_system(self):
        # TODO: Return A, B, Q, R for this system
        A = np.array([[1, self.dt], [15*self.dt, 1]])
        B = np.array([[0], [3*self.dt]])
        #Have the same cost function as the previous case 
        Q = np.array([[1,0], [0,1]])
        R = np.array([[1]])
        return A, B, Q, R
