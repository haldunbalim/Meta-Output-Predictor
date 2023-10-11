import numpy as np
import sympy


class DroneConfig:
    min_speed = -4
    max_speed = 4

    xmin = -20
    xmax = 20
    zmin = -20
    zmax = 20
    phimin = -np.pi/3
    phimax = np.pi/3
    vxmin = -2
    vxmax = 2
    vzmin = -2
    vzmax = 2
    phidotmin = -np.pi/4
    phidotmax = np.pi/4

    g = 10


class DroneState:
    limits = np.array([[DroneConfig.xmin, DroneConfig.zmin, DroneConfig.phimin, DroneConfig.vxmin, DroneConfig.vzmin, DroneConfig.phidotmin],
                       [DroneConfig.xmax, DroneConfig.zmax, DroneConfig.phimax, DroneConfig.vxmax, DroneConfig.vzmax, DroneConfig.phidotmax]], dtype="f")

    def __init__(self, x, z, phi, vx, vz, phidot):
        super().__init__()
        self.x = np.clip(x, *self.limits[:, 0])
        self.z = np.clip(z, *self.limits[:, 1])
        self.phi = np.clip(phi, *self.limits[:, 2])
        self.vx = np.clip(vx, *self.limits[:, 3])
        self.vz = np.clip(vz, *self.limits[:, 4])
        self.phidot = np.clip(phidot, *self.limits[:, 5])


    def __array__(self):
        return np.array([self.x, self.z, self.phi, self.vx, self.vz, self.phidot]).astype("f")

    def __repr__(self):
        return "x: {:.3f}, z:{:.3f} phi:{:.3f} vx:{:.3f} vz:{:.3f}, phiddot:{:.3f}".format(self.x, self.z, self.phidot, self.vx, self.vz, self.phiddot)


    def get_f(self):
        vx,vz,phi,phidot = self.vx,self.vz,self.phi,self.phidot
        g = DroneConfig.g
        f = np.array([vx*np.cos(phi) - vz*np.sin(phi),
                      vx*np.sin(phi)+ vz*np.cos(phi),
                      phidot,
                      vz*phidot-g*np.sin(phi),
                      -vx*phidot-g*np.cos(phi),
                      0], dtype="f")
        return f
    
    def get_g(self, m, l, J):
        g = np.zeros((6,2))
        g[4,:] = 1/m
        g[5,0] = l/J
        g[5,1] = -l/J
        return g

    @classmethod
    def generate_random(cls):
        return cls(*[np.random.randn()]*6)
    

class DroneAction:
    # w = v/r
    limits = np.array([[-0.5, -0.5],
                       [0.5, 0.5]], dtype="f")

    def __init__(self, u0, u1):
        super().__init__()
        self.u0 = np.clip(u0, *self.limits[:, 0])
        self.u1 = np.clip(u1, *self.limits[:, 1])

    def __repr__(self):
        return "{:.3f} {:.3f}".format(self.u0, self.u1)

    def __array__(self):
        return np.array([self.u0, self.u1]).astype(np.float32)

    @classmethod
    def generate_random(cls):
        return cls(*[np.random.rand()*(u-l)+l for l, u in cls.limits.T])


class DroneSim:
    def __init__(self, m=1, l=1, J=1, sigma_w=1e-1, sigma_v=1e-1, dt=0.1):
        self.m = m
        self.l = l
        self.J = J
        self.dt = dt
        self.sigma_w = sigma_w
        self.sigma_v = sigma_v
        self.C = np.random.rand(3, 6)

    def calculate_next_state(self, state, action):
        return DroneSim._calculate_next_state(state, action, self.m, self.l, self.J, self.dt, self.sigma_w)

    @staticmethod
    def _calculate_next_state(state, action, m, l, J, dt, sigma_w):
        f = state.get_f()
        g = state.get_g(m, l, J)
        nxt = np.array(state)+(f + g@np.array(action))*dt
        if sigma_w > 0:
            nxt += np.random.normal(scale=sigma_w, size=nxt.shape)
        return DroneState(*nxt)

    def simulate(self, traj_len):
        states = [DroneState.generate_random()]
        actions = []
        for _ in range(traj_len):
            u = DroneAction.generate_random()
            xn = self.calculate_next_state(states[-1], u)
            states.append(xn)
            actions.append(u)
        states = np.array([np.array(s) for s in states])
        actions = np.array([np.array(a) for a in actions])
        obs = states@self.C.T
        if self.sigma_v > 0:
            obs += np.random.normal(scale=self.sigma_v, size=(len(obs), 1))
        return states.astype("f"), obs.astype("f"), actions.astype("f")


def generate_drone_sample(n_positions, m_rng=(0.5, 2), l_rng=(0.5, 2), J_rng=(0.5, 2),  dt=1e-1, sigma_w=1e-1, sigma_v=1e-1):
    m = np.random.uniform(*m_rng)
    l = np.random.uniform(*l_rng)
    J = np.random.uniform(*J_rng)
    return _generate_drone_sample(n_positions, m, l, J, dt=dt, sigma_w=sigma_w, sigma_v=sigma_v)


def _generate_drone_sample(n_positions, m, l, J, dt=1e-1, sigma_w=1e-1, sigma_v=1e-1):
    # generate trajectory
    dsim = DroneSim(m=m, l=l, J=J, dt=dt, sigma_w=sigma_w, sigma_v=sigma_v)
    states, obs, actions = dsim.simulate(n_positions)
    return dsim, {"obs": obs, "states": states, "actions":actions}


def apply_ekf_drone(dsim, ys, us, sigma_w=None, sigma_v=None, x0=None, P0=None):
    C = dsim.C
    ny, nx = C.shape

    sigma_w = dsim.sigma_w if sigma_w is None else sigma_w
    sigma_v = dsim.sigma_v if sigma_v is None else sigma_v
    Q = np.eye(nx) * sigma_w ** 2
    R = np.eye(ny) * sigma_v ** 2
    g = DroneConfig.g

    x, z, phi, vx, vz, phidot = sympy.symbols('x z phi vx vz phidot')
    x = sympy.Matrix([x, z, phi, vx, vz, phidot])
    dfx = sympy.Matrix([vx*sympy.cos(phi) - vz*sympy.sin(phi),
                        vx*sympy.sin(phi) + vz*sympy.cos(phi),
                        phidot,
                        vz*phidot-g*sympy.sin(phi),
                        -vx*phidot-g*sympy.cos(phi),
                        0])
    fx = x + dfx * dsim.dt
    F = sympy.lambdify(x, fx.jacobian(x), 'numpy')

    P_k_km1 = np.eye(nx) if P0 is None else P0
    x_preds = [np.zeros(nx) if x0 is None else x0]

    for y, u in zip(ys, us):
        # update
        y_tilde_k = y - dsim.C @ x_preds[-1]
        S_k = C @ P_k_km1 @ C.T + R
        K_k = P_k_km1 @ C.T @ np.linalg.inv(S_k)
        x_k_k = x_preds[-1] + K_k @ y_tilde_k
        P_k_k = (np.eye(nx) - K_k @ C) @ P_k_km1

        F_curr = F(*x_k_k)
        # predict
        x_preds.append(np.array(dsim.calculate_next_state(DroneState(*x_k_k), DroneAction(*u))))
        P_k_km1 = F_curr @ P_k_k @ F_curr.T + Q

    return np.array(x_preds).astype("f") @ C.T
