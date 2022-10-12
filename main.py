import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

np.random.seed(1)
tf.set_random_seed(1)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, layers_U, layers_mu, xnu, xb, yb, F_val, f, num_train_tps):

        # Initialize NNs
        self.layers_U = layers_U
        self.weights_U, self.biases_U, self.adaps_U = self.initialize_NN(layers_U)
        self.layers_mu = layers_mu
        self.weights_mu, self.biases_mu, self.adaps_mu = self.initialize_NN(layers_mu)

        # Parameters
        self.xnu = xnu
        self.xb = xb
        self.yb = yb
        self.lb = np.array([xb[0], yb[0]])
        self.ub = np.array([xb[1], yb[1]])

        # Output file
        self.f = f
        
        # tf Placeholders
        # Points with Rigid Body Constraints
        self.x1_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.x2_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.y1_tf = tf.placeholder(tf.float32, shape = [1, 1])
        self.y2_tf = tf.placeholder(tf.float32, shape = [1, 1])

        # Coordinates of datapoints
        self.xx_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.yx_tf = tf.placeholder(tf.float32, shape=[None, 1])

        # Displacement Data
        self.ux_star_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.vx_star_tf = tf.placeholder(tf.float32, shape = [None, 1])

        # Test Points
        self.x_test_tf = tf.placeholder(tf.float32, shape = [None, 1])
        self.y_test_tf = tf.placeholder(tf.float32, shape = [None, 1])

        # Generate Training and Testing Points
        self.generateTrain(num_train_tps)

        # Physics
        # Interior (PDE)
        self.f1, self.f2, self.uI, self.vI, self.PxxI, self.PxyI, self.PyxI, self.PyyI, self.sxxI, self.sxyI, self.syxI, self.syyI, self.pI, self.JI, self.muI = self.pinn(self.xI, self.yI)
        # Left BC
        _, _, self.uL, self.vL, self.PxxL, self.PxyL, self.PyxL, self.PyyL, self.sxxL, self.sxyL, self.syxL, self.syyL, _, _, _ = self.pinn(self.xL, self.yL)
        # Right BC
        _, _, self.uR, self.vR, self.PxxR, self.PxyR, self.PyxR, self.PyyR, self.sxxR, self.sxyR, self.syxR, self.syyR, _, _, _ = self.pinn(self.xR, self.yR)
        # Top (Upper) BC
        _, _, self.uU, self.vU, self.PxxU, self.PxyU, self.PyxU, self.PyyU, self.sxxU, self.sxyU, self.syxU, self.syyU, _, _, _ = self.pinn(self.xU, self.yU)
        # Bottom (Lower) BC
        _, _, self.uD, self.vD, self.PxxD, self.PxyD, self.PyxD, self.PyyD, self.sxxD, self.sxyD, self.syxD, self.syyD, _, _, _ = self.pinn(self.xD, self.yD)
        # Rigid body motion
        _, _, self.u1, self.v1, _, _, _, _, _, _, _, _, _, _, _ = self.pinn(self.x1_tf, self.y1_tf)
        _, _, self.u2, self.v2, _, _, _, _, _, _, _, _, _, _, _ = self.pinn(self.x2_tf, self.y2_tf)

        # Data
        _, _, self.ux, self.vx, self.Pxxx, self.Pxyx, self.Pyxx, self.Pyyx, self.sxxx, self.sxyx, self.syxx, self.syyx, _, _, _ = self.pinn(self.xx_tf, self.yx_tf)

        # Test
        self.f1_test, self.f2_test, self.u_test, self.v_test, self.Pxx_test, self.Pxy_test, self.Pyx_test, self.Pyy_test, self.sxx_test, self.sxy_test, self.syx_test, self.syy_test, self.p_test, self.J_test, self.mu_test = self.pinn(self.x_test_tf, self.y_test_tf)
        
        # Loss
        # Interior (PDE)
        self.loss_I = (tf.reduce_mean(tf.square(self.f1)) + tf.reduce_mean(tf.square(self.f2)))
        # Right: tension; Left: displacement
        self.loss_L = tf.reduce_mean(tf.square(self.uL))    # + tf.reduce_mean(tf.square(self.PyxL))
        self.loss_R = tf.reduce_mean(tf.square(self.PxxR - F_val[0])) + tf.reduce_mean(tf.square(self.PyxR))
        # Up & Down: free
        self.loss_U = tf.reduce_mean(tf.square(self.PxyU)) + tf.reduce_mean(tf.square(self.PyyU - F_val[1]))
        self.loss_D = tf.reduce_mean(tf.square(self.PxyD)) + tf.reduce_mean(tf.square(self.PyyD - F_val[1]))
        # Rigid Body Motion
        self.loss_rigid = tf.square(tf.reduce_sum(self.v1))
        # Data points
        self.loss_x = tf.reduce_mean(tf.square(self.ux - self.ux_star_tf)) + tf.reduce_sum(tf.square(self.vx - self.vx_star_tf))
        # Incompressibility
        self.loss_IC = tf.reduce_mean(tf.square(self.JI - 1.0))
        # Total Loss
        self.loss = (self.loss_IC + self.loss_I * 1.0) + (self.loss_L + self.loss_R) * 1.0 + self.loss_U + self.loss_D + self.loss_rigid * 1.0  + self.loss_x * 10.0 

        # Optimizer
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        # tf session
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        adaps = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            a = tf.Variable(1.0, dtype=tf.float32)
            weights.append(W)
            biases.append(b)
            adaps.append(a)        
        return weights, biases, adaps
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def net_U(self, X):
        weights = self.weights_U
        biases = self.biases_U
        adaps = self.adaps_U
        num_layers = len(weights) + 1
        h = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a,tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        U = tf.add(tf.matmul(h, W), b)
        return U

    def net_mu(self, X):
        weights = self.weights_mu
        biases = self.biases_mu
        adaps = self.adaps_mu
        num_layers = len(weights) + 1
        h = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            a = adaps[l]
            h = tf.tanh(tf.multiply(a,tf.add(tf.matmul(h, W), b)))
        W = weights[-1]
        b = biases[-1]
        mu = tf.add(tf.matmul(h, W), b)
        return mu + 1.0/3      # shift towards positive values

    def pinn(self, x, y):

        X = tf.concat([x, y], 1)
        U = self.net_U(X)

        # Displacement and Pressure
        u = U[:, 0:1]
        v = U[:, 1:2]
        p = U[:, 2:3]

        # Material Properties
        mu = self.net_mu(X)

        # Deformation Gradient
        u_x = tf.gradients(u, x)[0] # du/dx
        u_y = tf.gradients(u, y)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]

        # First Piola-Kirchhoff Stress
        Pxx = -p * (v_y + 1) + mu * (u_x + 1)
        Pxy = p * v_x + mu * u_y
        Pyx = p * u_y + mu * v_x
        Pyy = -p * (u_x + 1) + mu * (v_y + 1)

        # Equilibrium Equation
        Pxx_x = tf.gradients(Pxx, x)[0] # dPxx/dx
        Pxy_y = tf.gradients(Pxy, y)[0]
        Pyx_x = tf.gradients(Pyx, x)[0]
        Pyy_y = tf.gradients(Pyy, y)[0]        
        f1 = Pxx_x + Pxy_y  # Residual of the equilibrium equation in x direction
        f2 = Pyx_x + Pyy_y

        # Cauchy Stress
        Sxx = Pxx * (u_x + 1) + Pxy * u_y
        Sxy = Pxx * v_x + Pxy * (v_y + 1)
        Syx = Pyx * (u_x + 1) + Pyy * u_y
        Syy = Pyx * v_x + Pyy * (v_y + 1)

        # Volume Change
        J = (u_x + 1) * (v_y + 1) - u_y * v_x

        return f1, f2, u, v, Pxx, Pxy, Pyx, Pyy, Sxx, Sxy, Syx, Syy, p, J, mu

    def generateTrain(self, num_line):
        x_1D = tf.linspace(np.float32(self.xb[0]), np.float32(self.xb[1]), num_line+1)
        y_1D = tf.linspace(np.float32(self.yb[0]), np.float32(self.yb[1]), num_line+1)
        x_2D = tf.multiply(x_1D[:,None], tf.ones([1, num_line+1]))
        y_2D = tf.multiply(tf.ones([num_line+1, 1]), y_1D[None,:])
        self.xI = tf.reshape(x_2D, [-1, 1])
        self.yI = tf.reshape(y_2D, [-1, 1])

        # Top boundary
        msize = (self.xb[1] - self.xb[0]) / num_line
        x_outer = tf.concat([[np.float32(self.xb[1])], tf.linspace(np.float32(self.xb[1] - msize/2), np.float32(self.xb[0] + msize/2) , num_line), [np.float32(self.xb[0])]], axis = 0)
        y_outer = self.yb[1] * tf.ones(x_outer.shape)
        self.xU = x_outer[:, None]
        self.yU = y_outer[:, None]

        # Left boundary
        msize = (self.yb[1] - self.yb[0]) / num_line
        y_outer = tf.concat([[np.float32(self.yb[1])], tf.linspace(np.float32(self.yb[1] - msize/2), np.float32(self.yb[0] + msize/2), num_line), [np.float32(self.yb[0])]], axis = 0)
        x_outer = self.xb[0] * tf.ones(y_outer.shape)    
        self.xL = x_outer[:, None]
        self.yL = y_outer[:, None]

        # Bottom boundary
        msize = (self.xb[1] - self.xb[0]) / num_line
        x_outer = tf.concat([[np.float32(self.xb[0])], tf.linspace(np.float32(self.xb[0] + msize/2), np.float32(self.xb[1] - msize/2) , num_line), [np.float32(self.xb[1])]], axis = 0)
        y_outer = self.yb[0] * tf.ones(x_outer.shape)
        self.xD = x_outer[:, None]
        self.yD = y_outer[:, None]

        # Right boundary
        msize = (self.yb[1] - self.yb[0]) / num_line  
        y_outer = tf.concat([[np.float32(self.yb[0])], tf.linspace(np.float32(self.yb[0] + msize/2), np.float32(self.yb[1] - msize/2) , num_line), [np.float32(self.yb[1])]], axis = 0)
        x_outer = self.xb[1] * tf.ones(y_outer.shape)
        self.xR = x_outer[:, None]
        self.yR = y_outer[:, None]
        return


    def train(self, Xx, Ux, it, printloss):

        # Data points
        xx = Xx[:, 0:1]
        yx = Xx[:, 1:2]
        ux = Ux[:, 0:1]
        vx = Ux[:, 1:2]

        # rigid body points
        x1 = np.array([[self.xb[0]]])
        x2 = np.array([[self.xb[1]]])
        y1 = np.array([[self.yb[0]]])
        y2 = np.array([[self.yb[0]]])

        tf_dict = { self.x1_tf: x1, self.y1_tf: y1,
                    self.x2_tf: x2, self.y2_tf: y2,
                    self.xx_tf: xx, self.yx_tf: yx,
                    self.ux_star_tf: ux, self.vx_star_tf: vx,
                    }

        self.sess.run(self.train_op_Adam, tf_dict)

        # Print   
        loss_value, loss_value_I, loss_value_L, loss_value_R, loss_value_U, loss_value_D, loss_value_rigid, loss_value_x, loss_value_IC = self.sess.run([self.loss, self.loss_I, self.loss_L, self.loss_R, self.loss_U, self.loss_D, self.loss_rigid, self.loss_x, self.loss_IC], tf_dict)
        loss_value_array = [loss_value_I, loss_value_L, loss_value_R, loss_value_U, loss_value_D, loss_value_rigid, loss_value_x, loss_value_IC]
        np.set_printoptions(precision=3)
        content = 'It: %d, Loss: %.3e' %(it, loss_value) +  '  Losses ILRUDrxIC:' + str(loss_value_array)
        print(content, flush = True)
        self.f.write(content + "\n")
        return loss_value, loss_value_array
        
    def test(self, num_line):
        x_1D = np.linspace(self.xb[0], self.xb[1], num_line+1)
        y_1D = np.linspace(self.yb[0], self.yb[1], num_line+1)
        x_2D = np.matmul(x_1D[:,None], np.ones((1, num_line+1)))
        y_2D = np.matmul(np.ones((num_line+1, 1)), y_1D[None,:])
        x_test = np.reshape(x_2D, [-1, 1])
        y_test = np.reshape(y_2D, [-1, 1])

        tf_dict = {self.x_test_tf: x_test, self.y_test_tf: y_test}
        u_test = self.sess.run(self.u_test, tf_dict)
        v_test = self.sess.run(self.v_test, tf_dict)
        Pxx_test = self.sess.run(self.Pxx_test, tf_dict)
        Pxy_test = self.sess.run(self.Pxy_test, tf_dict)
        Pyy_test = self.sess.run(self.Pyy_test, tf_dict)
        f1_test = self.sess.run(self.f1_test, tf_dict)
        f2_test = self.sess.run(self.f2_test, tf_dict)
        mu_test = self.sess.run(self.mu_test, tf_dict)
        X_test = np.hstack((x_test, y_test))
        U_test = np.hstack((u_test, v_test))
        return X_test, U_test, mu_test, f1_test, f2_test, Pxx_test, Pxy_test, Pyy_test

def import_data(filepath):
    # Read in data from FEM simulation
    data = np.load(filepath, encoding = 'latin1', allow_pickle = True)
    # Data on 21x21 points as data fed to the PINN
    Xx = data[0][:, 0:2]
    Ux = data[0][:, 2:]
    # Data on FEM nodes (19521 nodes) as the reference solution
    # Not fed to the PINN for training; used for validation of PINN efficacy only
    XFEM = data[1][:,1:3]
    UFEM = data[1][:,3:]
    return Xx, Ux, XFEM, UFEM

def mu_val(x, y):
    E = -0.15 * ((x+1)**2+(y+0.5)**2) + 1.0  + 0.4 * np.exp(-((x-0.1)**2+(y-0.2)**2)/2/(0.15)**2)
    mu = E / 3.0
    return mu

def ElasImag(nIter = 1000000, print_period = 1000, plot_period = 10000):

    # Geometry (domain bounds) and Meshing
    xb = np.array([-0.5, 0.5])
    yb = np.array([-0.5, 0.5])

    # Material properties
    xnu = 0.5

    # load value
    F_val = [0.3, 0.0]

    # Network Structure
    layers_U = [2, 30, 30, 30, 30, 3]
    layers_mu = [2, 30, 30, 30, 30, 1]

    # Generate training (informed) & true value
    Xx, Ux, XFEM, UFEM = import_data('data.npy')

    f = open("loss_record.txt","w")

    figure_path = './Figure/'
    if not os.path.isdir(figure_path):
        os.makedirs(figure_path)

    num_train_tps = 40
    num_test_tps = 80

    # Create the model
    model = PhysicsInformedNN(layers_U, layers_mu, xnu, xb, yb, F_val, f, num_train_tps)

    it_array = []
    loss_array = []
    losses_array = []
    data_save = {}

    start_time = time.time()

    for it in range(1, nIter+1):
        loss, losses = model.train(Xx, Ux, it, it%print_period==0)
        if (it%print_period==0):
            loss_array.append(loss)
            losses_array.append(losses)
            it_array.append(it)
            dt = time.time() - start_time
            print('Time: ', dt)
            start_time = time.time()
        if (it%plot_period==0 or it==print_period):
            X_test, U_test, mu_test, f1_test, f2_test, Pxx_test, Pxy_test, Pyy_test = model.test(num_test_tps)
            print("Result Plotted...")
            #plt.rcParams.update({'font.size': 14})
            #################### Deformation ####################
            plt.figure(1)
            figtopic = 'Displacement'
            ampli = 1.0
            x0 = X_test[:,0]+0*U_test[:,0]*ampli
            y0 = X_test[:,1]+0*U_test[:,1]*ampli
            xPINNs = X_test[:,0]+U_test[:,0]*ampli
            yPINNs = X_test[:,1]+U_test[:,1]*ampli
            xFEM = XFEM[:,0]+UFEM[:,0]*ampli
            yFEM = XFEM[:,1]+UFEM[:,1]*ampli
            plt.plot(x0, y0, '.k' , label = 'undeformed')
            plt.plot(xFEM, yFEM, 'xb', label = 'FEM')
            plt.plot(xPINNs, yPINNs, '.r' , label = 'deformed')
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            plt.legend()
            plt.title(figtopic)
            #plt.axis('equal')
            plt.xlim((-0.6, 1.0))
            plt.ylim((-0.6, 0.6))
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig1 = (x0, y0, xPINNs, yPINNs, xFEM, yFEM)

            plt.close(1)

            #################### U error colormap ##############
            plt.figure(101)
            figtopic = 'DispErr_x'
            num_dat = Xx.shape[0]
            num_dat_1D = np.int(np.round(np.sqrt(num_dat)))
            X_test2, U_test2, _, _, _, _, _, _ = model.test(num_dat_1D-1)
            seq_test2 = np.lexsort((X_test2[:,1],X_test2[:,0]))
            seq_x = np.lexsort((Xx[:,1], Xx[:,0]))
            x2D = np.reshape(X_test2[seq_test2,0:1], [num_dat_1D, num_dat_1D])
            y2D = np.reshape(X_test2[seq_test2,1:], [num_dat_1D, num_dat_1D])
            err_u = U_test2[seq_test2,0:1] - Ux[seq_x,0:1]
            err_u_2D = np.reshape(err_u, [num_dat_1D, num_dat_1D])
            cs = plt.contourf(x2D, y2D, err_u_2D, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(101)

            plt.figure(102)
            figtopic = 'DispErr_y'
            num_dat = Xx.shape[0]
            num_dat_1D = np.int(np.round(np.sqrt(num_dat)))
            X_test2, U_test2, _, _, _, _, _, _ = model.test(num_dat_1D-1)
            seq_test2 = np.lexsort((X_test2[:,1],X_test2[:,0]))
            seq_x = np.lexsort((Xx[:,1], Xx[:,0]))
            x2D = np.reshape(X_test2[seq_test2,0:1], [num_dat_1D, num_dat_1D])
            y2D = np.reshape(X_test2[seq_test2,1:], [num_dat_1D, num_dat_1D])
            err_v = U_test2[seq_test2,1:2] - Ux[seq_x,1:2]
            err_v_2D = np.reshape(err_v, [num_dat_1D, num_dat_1D])
            cs = plt.contourf(x2D, y2D, err_v_2D, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(102)

            data_fig101 = (x2D, y2D, U_test2[seq_test2], Ux[seq_x], err_u_2D, err_v_2D)

            #################### mu Profile ####################
            plt.figure(2)
            figtopic = 'Modulus'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            mu2D_PINNs = np.reshape(mu_test, [num_test_tps+1, num_test_tps+1])
            mu2D_FEM = np.reshape(mu_val(x2D, y2D), [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, mu2D_PINNs, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

            plt.close(2)

            #################### Err_mu Profile ####################
            plt.figure(3)
            figtopic = 'ModulusError'
            cs = plt.contourf(x2D, y2D, mu2D_PINNs-mu2D_FEM, 100)
            plt.colorbar(cs)
            #plt.clim(-0.2, 2.0)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig23 = (x2D, y2D, mu2D_PINNs, mu2D_FEM)

            plt.close(3)

            #################### loss ####################
            plt.figure(4)
            figtopic = 'loss'
            plt.semilogy(it_array, loss_array, '-b')
            plt.xlabel("Num Iteration")
            plt.ylabel("Loss")
            plt.title(figtopic)
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

            data_fig4 = (it_array, loss_array, losses_array)

            plt.close(4)

            #################### f1 Profile ####################
            plt.figure(5)
            figtopic = 'f1 Profile'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            f12D = np.reshape(f1_test, [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, f12D, 100)
            plt.colorbar(cs)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(5)

            #################### f2 Profile ####################
            plt.figure(6)
            figtopic = 'f2 Profile'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            f22D = np.reshape(f2_test, [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, f22D, 100)
            plt.colorbar(cs)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(6)

            data_fig56 = (x2D, y2D, f12D, f22D)

            #################### Pxx Profile ####################
            plt.figure(7)
            figtopic = 'Pxx Profile'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            Pxx2D = np.reshape(Pxx_test, [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, Pxx2D, 100)
            plt.colorbar(cs)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(7)

            #################### Pxy Profile ####################
            plt.figure(8)
            figtopic = 'Pxy Profile'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            Pxy2D = np.reshape(Pxy_test, [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, Pxy2D, 100)
            plt.colorbar(cs)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(8)

            #################### Pyy Profile ####################
            plt.figure(9)
            figtopic = 'Pyy Profile'
            x2D = np.reshape(X_test[:,0:1], [num_test_tps+1, num_test_tps+1])
            y2D = np.reshape(X_test[:,1:], [num_test_tps+1, num_test_tps+1])
            Pyy2D = np.reshape(Pyy_test, [num_test_tps+1, num_test_tps+1])
            cs = plt.contourf(x2D, y2D, Pyy2D, 100)
            plt.colorbar(cs)
            plt.title(figtopic)
            plt.axis('equal')
            plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')
            plt.close(9)

            data_fig789 = (x2D, y2D, Pxx2D, Pxy2D, Pyy2D)

            if (it==plot_period):   # Plot only once
                #################### mu True Profile ####################
                plt.figure(101)
                figtopic = 'ModulusTrue'
                cs = plt.contourf(x2D, y2D, mu2D_FEM, 100)
                plt.colorbar(cs)
                #plt.clim(-0.2, 2.0)
                plt.title(figtopic)
                plt.axis('equal')
                plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

                plt.close(101)

                ################### Training Points #################
                plt.figure(102)
                figtopic = 'Training Points'
                xI = model.sess.run(model.xI)
                yI = model.sess.run(model.yI)
                xL = model.sess.run(model.xL)
                yL = model.sess.run(model.yL)
                xR = model.sess.run(model.xR)
                yR = model.sess.run(model.yR)
                xU = model.sess.run(model.xU)
                yU = model.sess.run(model.yU)
                xD = model.sess.run(model.xD)
                yD = model.sess.run(model.yD)
                xB = np.vstack((xL, xR, xU, xD))
                yB = np.vstack((yL, yR, yU, yD))
                xx = Xx[:,0:1]
                yx = Xx[:,1:2]
                plt.plot(xI, yI,'.k', label = 'PDE')
                plt.plot(xB, yB, '.b', label = 'Boundary')
                plt.plot(xx, yx, 'xr', label = 'Data')
                plt.title(figtopic)
                plt.axis('equal')
                plt.savefig(figure_path + 'Fig_' + figtopic + '_It_' + str(it) + '.png')

                #data_fig8 = (xI, yI, xB, yB, xx, yx)

                plt.close(102)

            data_save[it] = (data_fig1, data_fig23, data_fig4, data_fig56, data_fig789,data_fig101)
            np.save('PINNs_Output.npy', data_save)

ElasImag(nIter = 2000000, print_period = 1000, plot_period = 10000)