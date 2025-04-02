import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_to_tensor
import scipy


class SOFIA:
    def __init__(self, R, m, lambda1, lambda2, lambda3, mu, phi, tol):
        self.R = R
        self.m = m
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.mu = mu
        self.phi = phi
        self.tol = tol


    @staticmethod
    def soft_thresholding(X, lambda3):
        return np.sign(X) * np.where(np.abs(X) - lambda3 < 0.0, 0.0, np.abs(X) - lambda3)
        

    def als(self, Y, max_iter=100):
        oldfit = 0.0
        UtU = [np.dot(u.T, u) for u in self.U]
        for iter in range(max_iter):
            # non-temporal mode
            for n1 in range(1, len(self.n_dims)):
                C = np.ones((1, self.R))
                B = np.ones((self.R, self.R))
                for n2 in range(len(self.n_dims)):
                    if n1 == n2:
                        continue
                    C = scipy.linalg.khatri_rao(C, self.U[n2])
                    B *= UtU[n2]

                self.U[n1] = np.dot(np.dot(tl.unfold(Y, n1), C), np.linalg.pinv(B))
                weight = tl.norm(self.U[n1], order=2, axis=0)
                self.U[n1] /= weight
                self.U[0] *= weight

                UtU[0] = np.dot(self.U[0].T, self.U[0])
                UtU[n1] = np.dot(self.U[n1].T, self.U[n1])

            # temporal mode
            C = np.ones((1, self.R))
            B = np.ones((self.R, self.R))
            for n in range(1, len(self.n_dims)):
                C = scipy.linalg.khatri_rao(C, self.U[n])
                B *= UtU[n]

            C = np.dot(tl.unfold(Y, 0), C)
            inv_B1 = np.linalg.pinv(B.copy() + (self.lambda1 + self.lambda2) * np.eye(self.R))
            inv_B2 = np.linalg.pinv(B.copy() + (2.0 * self.lambda1 + self.lambda2) * np.eye(self.R))
            inv_B3 = np.linalg.pinv(B.copy() + 2.0 * (self.lambda1 + self.lambda2) * np.eye(self.R))

            self.U[0][0] = np.dot(C[0] + self.lambda1 * self.U[0][1] + self.lambda2 * self.U[0][self.m], inv_B1)
            for i in range(1, self.m):
                self.U[0][i] = np.dot(C[i] + self.lambda1 * (self.U[0][i - 1] + self.U[0][i + 1]) + self.lambda2 * self.U[0][i + self.m], inv_B2)
            for i in range(self.m, self.n_dims[0] - self.m):
                self.U[0][i] = np.dot(C[i] + self.lambda1 * (self.U[0][i - 1] + self.U[0][i + 1]) + self.lambda2 * (self.U[0][i - self.m] + self.U[0][i + self.m]), inv_B3)
            for i in range(self.n_dims[0] - self.m, self.n_dims[0] - 1):
                self.U[0][i] = np.dot(C[i] + self.lambda1 * (self.U[0][i - 1] + self.U[0][i + 1]) + self.lambda2 * self.U[0][i - self.m], inv_B2)
            self.U[0][-1] = np.dot(C[-1] + self.lambda1 * self.U[0][-2] + self.lambda2 * self.U[0][-1 - self.m], inv_B1)

            UtU[0] = np.dot(self.U[0].T, self.U[0])

            fitness = 1 - tl.norm(Y - cp_to_tensor((np.ones(self.R), self.U)), order=2) / tl.norm(Y, order=2)
            if abs(oldfit - fitness) < self.tol:
                break
            oldfit = fitness

        print(f"fit: {fitness}")
        return cp_to_tensor((np.ones(self.R), self.U))


    def initialize(self, Y, max_epoch):
        self.n_dims = Y.shape
        lambda3 = self.lambda3
        lambda3_init = lambda3

        np.random.seed(1)
        rng = np.random.default_rng()
        self.U = [rng.random((n_dim, self.R)) for n_dim in self.n_dims]
        O = np.zeros(self.n_dims)

        for epoch in range(1, max_epoch + 1):
            Yo = Y - O
            X = self.als(Yo)
            O = self.soft_thresholding(Y - X, lambda3)

            lambda3 *= 0.85
            lambda3 = max(lambda3, lambda3_init / 100.0)

            if epoch > 1:
                if tl.norm(X_pre - X, order=2) / tl.norm(X_pre, order=2) < self.tol:
                    break
            X_pre = X

        self.alpha = np.zeros(self.R)
        self.beta = np.zeros(self.R)
        self.gamma = np.zeros(self.R)
        self.l = np.zeros(self.R)
        self.b = np.zeros(self.R)
        self.s = np.zeros((self.m, self.R))

        for r in range(self.R):
            '''def HW_fit_predict(alpha, beta, gamma, l0, b0, s0):
                l = np.zeros(self.n_dims[0] + self.m - 1)
                b = np.zeros(self.n_dims[0] + self.m - 1)
                s = np.zeros(self.n_dims[0] + self.m - 1)
                l[self.m - 1] = l0
                b[self.m - 1] = b0
                s[:self.m] = s0
                
                u = self.U[0][:, r]
    
                for t in range(self.m, self.n_dims[0] + self.m - 1):
                    l[t] = alpha * (u[t - (self.m - 1)] - s[t - self.m]) + (1.0 - alpha) * (l[t - 1] + b[t - 1])
                    b[t] = beta * (l[t] - l[t - 1]) + (1.0 - beta) * b[t - 1]
                    s[t] = gamma * (u[t - (self.m - 1)] - l[t - 1] - b[t - 1]) + (1.0 - gamma) * s[t - self.m]

                return l, b, s'''

            def HW_fit_predict(alpha, beta, gamma, l0, b0, s0):
                l = np.zeros(self.n_dims[0] + 1)
                b = np.zeros(self.n_dims[0] + 1)
                s = np.zeros(self.n_dims[0] + self.m)
                l[0] = l0
                b[0] = b0
                s[:self.m] = s0
                
                u = self.U[0][:, r]
    
                for t in range(1, self.n_dims[0] + 1):
                    l[t] = alpha * (u[t - 1] - s[t - 1]) + (1.0 - alpha) * (l[t - 1] + b[t - 1])
                    b[t] = beta * (l[t] - l[t - 1]) + (1.0 - beta) * b[t - 1]
                    s[t + self.m - 1] = gamma * (u[t - 1] - l[t - 1] - b[t - 1]) + (1.0 - gamma) * s[t - 1]

                return l, b, s

            def objective_func(params):
                alpha = params[0]
                beta = params[1]
                gamma = params[2]
                l0 = params[3]
                b0 = params[4]
                s0 = np.array(params[5:])
                l, b, s =  HW_fit_predict(alpha, beta, gamma, l0, b0, s0)
                #y = l[self.m - 1:] + b[self.m - 1:] + s[:self.n_dims[0]]
                y = l[:self.n_dims[0]] + b[:self.n_dims[0]] + s[:self.n_dims[0]]

                u = self.U[0][:, r]

                return tl.norm(u - y, order=2)

            init_alpha = 0.5 / self.m
            init_beta = 0.1 * init_alpha
            init_gamma = 0.05 * (1.0 - init_alpha)
            
            init_l0 = np.mean(self.U[0][:, r])
            init_b0 = np.mean((self.U[0][self.m:2*self.m, r] - self.U[0][:self.m, r]) / self.m)
            init_s0 = (self.U[0][:self.m, r] - init_l0).tolist()

            init_params = [init_alpha, init_beta, init_gamma, init_l0, init_b0] + init_s0
            bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (-1e3, 1e3), (-1e3, 1e3)] + [(-1e3, 1e3)] * self.m

            result = scipy.optimize.minimize(objective_func, init_params, method='L-BFGS-B', bounds=bounds)

            self.alpha[r] = result.x[0]
            self.beta[r] = result.x[1]
            self.gamma[r] = result.x[2]

            l, b, s = HW_fit_predict(self.alpha[r], self.beta[r], self.gamma[r], result.x[3], result.x[4], np.array(result.x[5:]))

            self.l[r] = l[-1]
            self.b[r] = b[-1]
            self.s[:, r] = s[-self.m:]

            #print(l)
            #print(b)
            #print(s); exit(0)

        self.sigma = self.lambda3 / 100.0 * np.ones(self.n_dims[1:])


    def dynamic_update(self, Yt):
        xt = self.HW_predict(1).reshape(self.R)
        Xt = cp_to_tensor((xt, self.U[1:]))
        YXt = Yt - Xt
        #print(Yt)
        #print(Xt);exit(0)

        # Estimate Outliers
        k = 2.0
        Ot = YXt - np.clip(YXt / self.sigma, a_min=-k, a_max=k) * self.sigma
        Rt = YXt - Ot
        #print(Rt)
        #print(Ot)

        # Update sigma
        ck = 2.52
        self.sigma *= np.sqrt(self.phi * np.where(np.abs(YXt / self.sigma) < k, ck * (1.0 - (1.0 - (YXt / self.sigma / k) ** 2.0) ** 3.0), ck) + (1.0 - self.phi))

        G = []
        weights = []

        # Update temporal factor
        C = np.ones((1, self.R))
        for n in range(1, len(self.n_dims)):
            C = scipy.linalg.khatri_rao(C, self.U[n])
        G.append(self.mu * (np.dot(C.T, Rt.reshape(-1)) + self.lambda1 * self.U[0][-1] + self.lambda2 * self.U[0][-self.m] - (self.lambda1 + self.lambda2) * xt))

        # Update non-temporal factors
        for n1 in range(1, len(self.n_dims)):
            C = xt.copy().reshape(1, self.R)
            for n2 in range(1, len(self.n_dims)):
                if n1 == n2:
                    continue
                C = scipy.linalg.khatri_rao(C, self.U[n2])

            G.append(self.mu * np.dot(tl.unfold(Rt, n1 - 1), C))
            weight = tl.norm(self.U[n1], order=2, axis=0)
            weights.append(weight)

        for n in range(len(self.n_dims)):
            self.U[n] += self.mu * G[n]

        for n in range(1, len(self.n_dims)):
            self.U[n] /= weights[n - 1]
            xt *= weights[n - 1]

        self.U[0][:-1] = self.U[0][1:]
        self.U[0][-1] = xt

        # Update HW parameters
        new_l = self.alpha * (xt - self.s[0]) + (1.0 - self.alpha) * (self.l + self.b)
        new_b = self.beta * (new_l - self.l) + (1.0 - self.beta) * self.b
        new_s = self.gamma * (xt - self.l - self.b) + (1.0 - self.gamma) * self.s[0]

        self.l = new_l
        self.b = new_b
        self.s[:-1] = self.s[1:]
        self.s[-1] = new_s


    def HW_predict(self, ls):
        x = np.zeros((ls, self.R))
        for h in range(ls):
            x[h] = self.l + (h + 1) * self.b + self.s[h]
        return x

    def predict(self, ls):
        return cp_to_tensor((np.ones(self.R), [self.HW_predict(ls)] + self.U[1:]))