# ------------------------------------------------------------------------------
# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/ConvSSM/blob/main/LICENSE
#
# Written by Jimmy Smith
# ------------------------------------------------------------------------------

from functools import partial
import jax
import jax.numpy as np
from flax import linen as nn
from jax.nn.initializers import he_normal, normal
from jax.numpy.linalg import eigh
from jax.scipy.linalg import block_diag

from . import diagonal_scans
from .conv_ops import VmapResnetBlock, VmapDiag_CD_Conv, VmapDiagResnetBlock, Half_GLU



def make_Q_skew(size, num_dim = 2):
    '''
    size: size of model input
    num_dim: dim of model input

    return: eigenvector of BTST (HW, HW)
    '''
    if num_dim == 1:
        N = size
        return np.array([[(1j)**(i+1)*np.sin(np.pi*(i+1)*(j+1)/(N+1))/np.sqrt((N+1)/2) for j in range(N)] for i in range(N)])
    if num_dim == 2:
        N, M = size
        x_matrix = np.array([[(1j)**(i+1)*np.sin(np.pi*(i+1)*(j+1)/(N+1))/np.sqrt((N+1)/2) for j in range(N)] for i in range(N)])
        y_matrix = np.array([[(1j)**(i+1)*np.sin(np.pi*(i+1)*(j+1)/(M+1))/np.sqrt((M+1)/2) for j in range(M)] for i in range(M)])
        return x_matrix, y_matrix



def make_basis_skew(size, num_dim = 2):
    '''
    size: size of model input
    num_dim: dim of model input

    return: basis of BTST (HW)
    '''
    if num_dim == 1:
        N = size
        b_temp = np.array([2*np.cos(np.pi*(i+1)/(N+1)) for i in range(N)])
        #a_temp = np.array([1 for i in range(N)], dtype = torch.float32)
        #return torch.stack((a_temp, b_temp), dim=-1)
        return b_temp.unsqueeze(-1)
    elif num_dim == 2:
        N, M = size
        d_temp = np.array([[4*np.cos(np.pi*(i+1)/(N+1))*np.cos(np.pi*(j+1)/(M+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        c_temp = np.array([[2*np.cos(np.pi*(i+1)/(N+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        b_temp = np.array([[2*np.cos(np.pi*(j+1)/(M+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        #a_temp = np.array([[1 for j in range(M)] for i in range(N)], dtype = torch.complex64, device = device).reshape(-1)
        #return torch.stack((a_temp, b_temp, c_temp, d_temp), dim=-1)
        return np.stack((b_temp, c_temp, d_temp), axis=-1)


def make_Q_sym(size, num_dim = 2):
    '''
    size: size of model input
    num_dim: dim of model input

    return: eigenvector of BTST (HW, HW)
    '''
    if num_dim == 1:
        N = size
        return np.array([[np.sin(np.pi*(i+1)*(j+1)/(N+1))/np.sqrt((N+1)/2) for j in range(N)] for i in range(N)])
    if num_dim == 2:
        N, M = size
        x_matrix = np.array([[np.sin(np.pi*(i+1)*(j+1)/(N+1))/np.sqrt((N+1)/2) for j in range(N)] for i in range(N)])
        y_matrix = np.array([[np.sin(np.pi*(i+1)*(j+1)/(M+1))/np.sqrt((M+1)/2) for j in range(M)] for i in range(M)])
        return x_matrix, y_matrix


def make_Q_identity(size, num_dim = 2):
    '''
    size: size of model input
    num_dim: dim of model input

    return: eigenvector of BTST (HW, HW)
    '''
    if num_dim == 1:
        N = size
        return np.identity(N)
    if num_dim == 2:
        N, M = size
        x_matrix = np.identity(N)
        y_matrix = np.identity(M)
        return np.array(x_matrix), np.array(y_matrix)


def make_basis_sym(size, num_dim = 2):
    '''
    size: size of model input
    num_dim: dim of model input

    return: basis of BTST (HW)
    '''
    if num_dim == 1:
        N = size
        b_temp = np.array([2*np.cos(np.pi*(i+1)/(N+1)) for i in range(N)])
        #a_temp = np.array([1 for i in range(N)], dtype = torch.complex64, device=device)
        #return torch.stack((a_temp, b_temp), dim=-1)
        return b_temp.unsqueeze(-1)
    elif num_dim == 2:
        N, M = size
        d_temp = np.array([[4*np.cos(np.pi*(i+1)/(N+1))*np.cos(np.pi*(j+1)/(M+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        c_temp = np.array([[2*np.cos(np.pi*(i+1)/(N+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        b_temp = np.array([[2*np.cos(np.pi*(j+1)/(M+1)) for j in range(M)] for i in range(N)]).reshape(-1)
        #a_temp = np.array([[1 for j in range(M)] for i in range(N)], dtype = torch.complex64, device = device).reshape(-1)
        #return torch.stack((a_temp, b_temp, c_temp, d_temp), dim=-1)
        return np.stack((b_temp, c_temp, d_temp), axis=-1)


def value_initializer(ssm_size, mini = False):
    if mini:
        size = (4,)
    else:
        size = (ssm_size, 4)
    return np.zeros(size)


################################


def make_HiPPO(N):
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_Normal_HiPPO(N):
    """normal approximation to the HiPPO-LegS matrix"""
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)
    nhippo = hippo + P[:, np.newaxis] * P[np.newaxis, :]

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)

    return nhippo, P, B


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


def discretize_zoh(Lambda, D, Delta, input_size):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            D (complex64): diagonalized BTST                       (P, HW)
            Delta (float32): discretization step sizes             (P,)
            input_size (tuple): H, W
        Returns:
            discretized A_bar (=total diag term) (complex64), B_coeff (complex64)  (H, W, P), (H, W, P)
    """
    Identity = np.ones(Lambda.shape[0])
    temp = Lambda[:, None] * D
    Lambda_bar = np.exp(temp * Delta[:, None])
    B_coeff = (1/temp * (Lambda_bar-Identity[:,None]))
    return np.reshape(np.swapaxes(Lambda_bar, 0, 1), (*input_size, -1)), np.reshape(np.swapaxes(B_coeff, 0, 1), (*input_size, -1))


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    U, dt_min, dt_max = input
    log_steps = []
    for i in range(U):
        key, skey = jax.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def initialize_C_kernel(key, shape):
    """For general kernels, e.g. C,D, encoding/decoding"""
    out_dim, in_dim, k = shape
    fan_in = in_dim*(k**2)

    # Note in_axes should be the first by default:
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
    return he_normal()(key,
                       (fan_in, out_dim)).reshape(out_dim,
                                                  in_dim,
                                                  k, k).transpose(0, 2, 3, 1).reshape(-1, in_dim)


def initialize_B_kernel(key, shape):
    """We will store the B kernel as a matrix,
    returns shape: (out_dim, in_dim*k*k)"""
    out_dim, in_dim, k = shape
    fan_in = in_dim*(k**2)

    # Note in_axes should be the first by default:
    # https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html#jax.nn.initializers.variance_scaling
    return he_normal()(key,
                       (fan_in, out_dim)).T


def init_VinvB(key, shape, Vinv):
    B = initialize_B_kernel(key, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def init_CV(key, shape, V):
    out_dim, in_dim, k = shape
    C = initialize_C_kernel(key, shape)
    CV = C @ V
    CV = CV.reshape(out_dim, k, k, in_dim//2).transpose(1, 2, 3, 0)
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)

################################
def get_kernel_values(values, mini):
    values = nn.softmax(values)*4
    if mini:
        x, y, z, w = values[0], values[1], values[2], values[3]
    else:
        x, y, z, w = values[:,0], values[:,1], values[:,2], values[:,3]
    a = (x+y-2)/4
    b = (x+z-2)/4
    c = (x+w-2)/8
    return np.stack((a, b, c), axis= -1)

def get_D(values, mini, basis, P):
    values = get_kernel_values(values, mini).astype(np.complex64)
    if mini:
        D = np.einsum('n, hn -> h', values, basis) + 1
        D = np.tile(D[None, :], (P, 1))
    else:
        D = np.einsum('pn,hn->ph', values, basis) + 1
    return D



class BTSTSSM(nn.Module):
    Lambda_re_init: np.array
    Lambda_im_init: np.array
    V: np.array
    Vinv: np.array
    clip_eigs: bool
    parallel: bool  # Compute scan in parallel
    activation: nn.module
    num_groups: int

    U: int    # Number of SSM input and output features
    P: int    # Number of state features of SSM
    input_size: tuple   # size of input image
    k_B: int  # B kernel width/height
    k_C: int  # C kernel width/height
    k_D: int  # D kernel width/height

    dt_min: float  # for initializing discretization step
    dt_max: float
    C_D_config: str = "standard"
    squeeze_excite: bool = False

    basis_type: str = 'skew'
    mini: bool = False

    def setup(self):
        # Initialize diagonal state to state transition kernel Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im
        
        ########################################
        if self.basis_type == 'skew':
            Q_h_skew, Q_w_skew = make_Q_skew(self.input_size, num_dim=2)
            self.Q_h = Q_h_skew
            self.Q_w = Q_w_skew
            self.basis = make_basis_skew(self.input_size, num_dim=2)
        elif self.basis_type == 'sym':
            Q_h_sym, Q_w_sym = make_Q_sym(self.input_size, num_dim=2)
            self.Q_h = Q_h_sym
            self.Q_w = Q_w_sym
            self.basis = make_basis_sym(self.input_size, num_dim=2)
        else:
            print('basis type not implimentated')
        
        self.values = self.param("values", lambda rng, shape: value_initializer(self.P, self.mini), (None, ))

        ########################################

        # Initialize input to state (B) and output to state (C) kernels
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(rng,
                                                          shape,
                                                          self.Vinv),
                            (2*self.P, self.U, self.k_B))
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]
        self.B_tilde = B_tilde.reshape(self.P, self.U, self.k_B, self.k_B).transpose(2, 3, 1, 0)

        self.C = self.param("C",
                            lambda rng, shape: init_CV(rng, shape, self.V),
                            (self.U, 2*self.P, self.k_C))
        self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        if self.C_D_config == "standard":
            self.C_D_conv = VmapDiag_CD_Conv(activation=self.activation,
                                             k_D=self.k_D,
                                             out_channels=self.U,
                                             num_groups=self.num_groups,
                                             squeeze_excite=self.squeeze_excite)
        elif self.C_D_config == "resnet":
            self.C_D_conv = VmapResnetBlock(activation=self.activation,
                                            k_size=self.k_D,
                                            out_channels=self.U,
                                            num_groups=self.num_groups,
                                            squeeze_excite=self.squeeze_excite)
        elif self.C_D_config == "diag_resnet":
            self.C_D_conv = VmapDiagResnetBlock(activation=self.activation,
                                                k_size=self.k_D,
                                                out_channels=self.U,
                                                num_groups=self.num_groups,
                                                squeeze_excite=self.squeeze_excite)

        elif self.C_D_config == "half_glu":
            self.C_D_conv = Half_GLU(dim=self.U)

        # Initialize learnable discretization steps
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = np.exp(self.log_step[:, 0])

        self.D = get_D(self.values, self.mini, self.basis, self.P)

        if self.parallel:
            # Discretize
            self.A_bar, self.B_coeff = discretize_zoh(self.Lambda,
                                                    self.D,
                                                    step,
                                                    self.input_size)
        else:
            # trick to cache the discretization for step-by-step
            # generation
            def init_discrete():
                A_bar, B_coeff = discretize_zoh(self.Lambda,
                                              self.D,
                                              step,
                                              self.input_size)
                return A_bar, B_coeff
            ssm_var = self.variable("prime", "ssm", init_discrete)
            if self.is_mutable_collection("prime"):
                ssm_var.value = init_discrete()
            self.ssm = ssm_var.value
    

    def __call__(self, input_sequence, x0):
        """
        input sequence is shape (L, bsz, H, W, U)
        x0 is (bsz, H, W, U)
        Returns:
            x_L (float32): the last state of the SSM  (bsz, H, W, P)
            ys (float32): the conv SSM outputs       (L,bsz, H, W, U)
        """
        if self.parallel:
            # TODO: right now parallel version assumes x_init is zeros
            x_last, ys = diagonal_scans.apply_convSSM_parallel(self.A_bar,
                                                               self.B_coeff,
                                                               self.B_tilde,
                                                               self.C_tilde,
                                                               input_sequence,
                                                               x0,
                                                               self.Q_h,
                                                               self.Q_w)

        else:
            # For sequential generation (e.g. autoregressive decoding)
            x_last, ys = diagonal_scans.apply_convSSM_sequential(*self.ssm,
                                                                 self.B_tilde,
                                                                 self.C_tilde,
                                                                 input_sequence,
                                                                 x0,
                                                                 self.Q_h,
                                                                 self.Q_w)
        if self.C_D_config == "standard":
            ys = self.C_D_conv(ys, input_sequence)
        elif self.C_D_config == "resnet":
            ys = self.C_D_conv(ys)
        elif self.C_D_config in ["half_glu"]:
            ys = jax.vmap(self.C_D_conv)(ys)
        return x_last, ys


def hippo_initializer(ssm_size, blocks):
    block_size = int(ssm_size/blocks)
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    ssm_size = ssm_size // 2
    block_size = block_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    Lambda = (Lambda * np.ones((blocks, block_size))).ravel()
    V = block_diag(*([V] * blocks))
    Vinv = block_diag(*([Vc] * blocks))

    return Lambda.real, Lambda.imag, V, Vinv, ssm_size


def init_BTSTSSM(ssm_size,
                   blocks,
                   clip_eigs,
                   U,
                   k_B,
                   k_C,
                   k_D,
                   dt_min,
                   dt_max,
                   C_D_config,
                   input_size,
                   basis_type,
                   mini):
    Lambda_re_init, Lambda_im_init,\
        V, Vinv, ssm_size = hippo_initializer(ssm_size, blocks)

    return partial(BTSTSSM,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   clip_eigs=clip_eigs,
                   U=U,
                   P=ssm_size,
                   input_size=input_size,
                   k_B=k_B,
                   k_C=k_C,
                   k_D=k_D,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   C_D_config=C_D_config,
                   basis_type=basis_type,
                   mini=mini
                   )
