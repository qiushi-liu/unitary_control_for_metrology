# Example in the main text

from algorithm import *

Pauli_X = np.array([[0, 1], [1, 0]])
Pauli_Y = np.array([[0, -1j], [1j, 0]])
Pauli_Z = np.array([[1, 0], [0, -1]])

IZ = np.kron(np.eye(2), Pauli_Z)
ZI = np.kron(Pauli_Z, np.eye(2))
XX = np.kron(Pauli_X, Pauli_X)
YY = np.kron(Pauli_Y, Pauli_Y)
ZZ = np.kron(Pauli_Z, Pauli_Z)
XY = np.kron(Pauli_X, Pauli_Y)

theta = 1.0
t=1.0
p1=0.1
p2=0.1

H0 = ZI + IZ + theta*(XX+YY+ZZ)
U_t = scipy.linalg.expm(-1j*t*H0)

# Kraus operators of the channel to estimate
def K_thetas_ZZ(theta, t, p1, p2):
    K1 = np.sqrt(1-p1-p2)*U_t
    K2 = np.sqrt(p1)*XX @ U_t
    K3 = np.sqrt(p2)*XY @ U_t
    return [K1, K2, K3]
    
# The derivative of Kraus operators of the channel to estimate
def dK_thetas_ZZ(theta, t, p1, p2):
    dU_t = U_t @ (-1j*t*(XX + YY + ZZ))
    dK1 = np.sqrt(1-p1-p2)*dU_t
    dK2 = np.sqrt(p1)*XX @ dU_t
    dK3 = np.sqrt(p2)*XY @ dU_t
    return [dK1, dK2, dK3]
    
K_thetas = K_thetas_ZZ(theta, t, p1, p2)
dK_thetas = dK_thetas_ZZ(theta, t, p1, p2)

# Transition matrix T_theta and its derivative dT_theta
T_theta = np.kron(K_thetas[0], K_thetas[0].conj()) + np.kron(K_thetas[1], K_thetas[1].conj()) + np.kron(K_thetas[2], K_thetas[2].conj())
dT_theta = (np.kron(dK_thetas[0], K_thetas[0].conj()) + np.kron(K_thetas[0], dK_thetas[0].conj())
            + np.kron(dK_thetas[1], K_thetas[1].conj()) + np.kron(K_thetas[1], dK_thetas[1].conj())
            + np.kron(dK_thetas[2], K_thetas[2].conj()) + np.kron(K_thetas[2], dK_thetas[2].conj()))

v1=np.reshape(np.linalg.eigh(T_theta.conj().T@T_theta)[1][:,-1], (16, 1))
v2=np.reshape(np.linalg.eigh(T_theta.conj().T@T_theta)[1][:,-2], (16, 1))
v3=np.reshape(np.linalg.eigh(T_theta.conj().T@T_theta)[1][:,-3], (16, 1))
v4=np.reshape(np.linalg.eigh(T_theta.conj().T@T_theta)[1][:,-4], (16, 1))
Proj = v1@v1.conj().T + v2@v2.conj().T + v3@v3.conj().T + v4@v4.conj().T

# R0 in the algorithm
R0 = np.reshape(np.linalg.eig(Proj@T_theta.conj().T@dT_theta@Proj)[1][:,1], (4, 4))
R0_out = np.reshape(T_theta @ np.reshape(R0, (16, 1)), (4, 4))

H1 = R0 + R0.conj().T
H2 = 1j*(R0 - R0.conj().T)
H1_out = R0_out + R0_out.conj().T
H2_out = 1j*(R0_out - R0_out.conj().T)

# List_in and List_out
H_in_list = [H1, H2]
H_out_list = [H1_out, H2_out]

R0_out = np.reshape(T_theta @ np.reshape(R0, (16, 1)), (4, 4))
result = unitary_control(H_in_list, H_out_list)

# output the unitary control and check the partially reversible condition
print('Control unitary: ', result)
print('Check the partially reversible condition (the difference should be 0): ', np.linalg.norm(result.conj().T @ R0 @ result - R0_out))
