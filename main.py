import numpy as np
import matplotlib.pyplot as plt

#Define the functions
def f1(xx, tt):
    y_1 = 2 * np.cos(xx) * np.exp(1j * tt)
    return y_1

def f2(xx, tt):
    y_2 = np.sin(xx) * np.exp(3j * tt)
    return y_2

#Define time and space discretizations
xi = np.linspace(-10, 10, 401)
t = np.linspace(0, 15, 201)
dt = t[1] - t[0]
tt, xx = np.meshgrid(t,xi)
X = f1(xx, tt) + f2(xx, tt)

print(X.shape)

plt.figure(figsize=(6, 4))
plt.contourf(tt, xx, np.real(X), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('t')
plt.ylabel('x')
plt.title('Contour plot of X')
plt.show()

X_1 = X[:, :-1]
X_2 = X[:, 1:]

print(X_1.shape, X_2.shape)

U, S, VT = np.linalg.svd(X_1,full_matrices=0)
V=VT.conj().T
S=np.diag(S)
print(U.shape,S.shape, V.shape)

plt.figure(figsize=(4, 4))
plt.plot(U[:, 0], label='U[:, 0]')
plt.plot(U[:, 1], label='U[:, 1]')
plt.legend(loc='upper left')
plt.show()

plt.figure(figsize=(4, 4))
plt.plot(np.diag(S[:10]), 'o-')
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('First 10 Singular Values of X')
plt.show()

print(np.diag(S[:4]))

plt.figure(figsize=(4, 4))
plt.plot(V[:,0], label='V[:,0]')
plt.plot(V[:,1], label='V[:,1]')
plt.legend(loc='upper left')
plt.show()

r =2
Ur = U[:,:r]
Sr = S[:r,:r]
Vr = V[:,:r]
print(Ur.shape, Sr.shape, Vr.shape)

print(X_2.shape)

print(Ur[:2,:2])

A_tilde =  (Ur.conj().T) @ X_2 @ Vr @ np.linalg.inv(Sr)

Lambda, W = np.linalg.eig(A_tilde)

#Lambda = np.diag(Lambda)
print(Lambda)

# Plot the eigenvalues in the complex plane
plt.figure(figsize=(4, 4))
plt.scatter(Lambda.real, Lambda.imag)

# Plot unit circle
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(np.cos(theta), np.sin(theta), linestyle='--', color='r', label='Unit Circle')

plt.axis('equal')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Complex Eigenvalues')
plt.legend(loc='upper left')
plt.show()

print(W)

Phi = X_2 @ Vr @ np.linalg.inv(Sr) @ W

print(Phi[:2,:])
Omega = np.log(Lambda)/dt
print(Omega.shape, Omega)
print("*******************")
print("Notice the imaginary parts", np.imag(Omega))

amp = np.linalg.lstsq(Phi,X_1[:,0],rcond=None)[0]

print(amp.shape[0], amp)

t_exp = np.arange(X.shape[1]) * dt
temp = np.repeat(Omega.reshape(-1,1), t_exp.size, axis=1)
dynamics = np.exp(temp * t_exp) * amp.reshape(amp.shape[0], -1)
print(t_exp.shape, temp.shape, dynamics.shape)
print(X.shape[1])
print(t_exp.size)

plt.figure(figsize=(4, 4))
plt.plot(t_exp, dynamics[0, :], '-', label='dynamics[0, :]')
plt.plot(t_exp, dynamics[1, :], '-', label='dynamics[1, :]')
plt.xlabel('t')
plt.ylabel('Dynamics')
plt.legend()
plt.title('Dynamics of the reduced model')
plt.show()

X_dmd = Phi @ dynamics
print(X_dmd.shape)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.contourf(tt, xx, np.real(X), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of X')

plt.subplot(1, 3, 2)
plt.contourf(tt, xx, np.real(X_dmd), 20, cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of X_dmd')

X_diff = np.real(X) - np.real(X_dmd)
plt.subplot(1, 3, 3)
plt.contourf(tt, xx, X_diff , cmap='RdGy')
plt.colorbar()
plt.xlabel('x')
plt.ylabel('t')
plt.title('Contour plot of Error')


plt.show()
