import sys
import json
from time import time
from . import lap_solvers

import numpy as np
from scipy import sparse


class _BaseSGM:
    def __init__(self, A, B, P, verbose=True):
        self.A = A
        self.B = B
        self.P = P

        self.verbose = verbose

    def _reset_timers(self):
        self.lap_times   = []
        self.iter_times  = []

    def _log_times(self):
        if self.verbose:
            print(json.dumps({
                "iter"      : len(self.lap_times),
                "lap_time"  : float(self.lap_times[-1]),
                "iter_time" : float(self.iter_times[-1]),
            }))

    def check_convergence(self, c, d, e, tolerance):
        cde = c + e - d
        d2e = d - 2 * e

        if (cde == 0) and (d2e == 0):
            alpha = 0
            falpha = -1 # NA value
        else:
            if (cde == 0):
                alpha  = float('inf')
                falpha = -1 # NA value
            else:
                alpha = -d2e / (2 * cde)
                falpha = cde * alpha ** 2 + d2e * alpha

        f1 = c - e

        if (alpha > 0) and (alpha < tolerance) and (falpha > max(0, f1)):
            return alpha, False # P <- (alpha * P) + (1 - alpha) * T
        elif f1 < 0:
            return None, False # P <- T
        else:
            return None, True  # stop

class _JVMixin:
    def __init__(self, *args, jv_backend='gatagat', **kwargs):
        # print('jv_backend=%s' % jv_backend, file=sys.stderr)
        self.jv_backend = jv_backend
        super().__init__(*args, **kwargs)

class BaseSGMClassic(_BaseSGM):
    def run(self, num_iters, tolerance, verbose=True):
        A, B, P = self.A, self.B, self.P
        if hasattr(self, '_warmup'):
            self._warmup(A, P, B)

        self._reset_timers()

        grad = self.compute_grad(A, P, B)

        for i in range(num_iters):
            iter_t = time()

            lap_t = time()
            T = self.solve_lap(grad)
            self.lap_times.append(time() - lap_t)

            gradt = self.compute_grad(A, T, B)

            ps_grad_P  = self.compute_trace(grad, P)
            ps_grad_T  = self.compute_trace(grad, T)
            ps_gradt_P = self.compute_trace(gradt, P)
            ps_gradt_T = self.compute_trace(gradt, T)

            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )

            if not stop:
                if alpha is not None:
                    P    = (alpha * P)    + (1 - alpha) * T
                    grad = (alpha * grad) + (1 - alpha) * gradt
                else:
                    P    = T
                    grad = gradt

            self.iter_times.append(time() - iter_t)
            if verbose:
                self._log_times()

            if stop:
                break

        return self.solve_lap(P, final=True)

class BaseSGMSparse(_BaseSGM):
    def run(self, num_iters, tolerance, verbose=True):
        A, B, P = self.A, self.B, self.P
        if hasattr(self, '_warmup'):
            self._warmup()
        
        self._reset_timers()
        
        
        AP   = A.dot(P)
        grad = AP.dot(B)
        
        for i in range(num_iters):
            iter_t = time()
            
            lap_t = time()
            rowcol_offsets = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
            T = self.solve_lap(grad, rowcol_offsets)
            self.lap_times.append(time() - lap_t)
            
            AT    = A.dot(T)
            gradt = AT.dot(B)
            
            ps_grad_P  = self.compute_trace(AP, B, P)
            ps_grad_T  = self.compute_trace(AP, B, T)
            ps_gradt_P = self.compute_trace(AT, B, P)
            ps_gradt_T = self.compute_trace(AT, B, T)
            
            alpha, stop = self.check_convergence(
                c=ps_grad_P,
                d=ps_gradt_P + ps_grad_T,
                e=ps_gradt_T,
                tolerance=tolerance,
            )
            
            if not stop:
                if alpha is not None:
                    P    = (alpha * P)    + (1 - alpha) * T
                    grad = (alpha * grad) + (1 - alpha) * gradt
                    AP   = (alpha * AP)   + (1 - alpha) * AT
                else:
                    P    = T
                    grad = gradt
                    AP   = AT
            
            self.iter_times.append(time() - iter_t)
            if verbose:
                self._log_times()
            
            if stop:
                break
        
        return self.solve_lap(P, None, final=True)

class _ScipySGMClassic(BaseSGMClassic): # BaseSGMSparse
    def compute_grad(self, A, P, B):
        AP = A.dot(P)
        sparse_part = 4 * AP.dot(B)
        dense_part  = - 2 * AP.sum(axis=1) - 2 * B.sum(axis=0) + A.shape[0]
        return np.asarray(sparse_part + dense_part)

    def compute_trace(self, x, y):
        return y.multiply(x).sum()

class ScipyJVClassicSGM(_JVMixin, _ScipySGMClassic):
    def solve_lap(self, cost, final=False):
        idx = lap_solvers.jv(cost, jv_backend=self.jv_backend)
        if final:
            return idx

        return sparse.csr_matrix((np.ones(cost.shape[0]), (np.arange(cost.shape[0]), idx)))



# --

class _ScipySGMSparse(BaseSGMSparse):
    def _warmup(self):
        cost = sparse.random(100, 100, density=0.5).tocsr()
        _ = self.solve_lap(cost, None)
    
    def compute_trace(self, AX, B, Y):
        YBt = Y.dot(B.T)
        
        AX_sum = Y.dot(AX.sum(axis=1)).sum()
        B_sum  = Y.T.dot(B.sum(axis=0).T).sum()
        
        return 4 * AX.multiply(YBt).sum() + AX.shape[0] * Y.sum() - 2 * (AX_sum + B_sum)


class JVSparseSGM(_JVMixin, _ScipySGMSparse):
    def solve_lap(self, cost, rowcol_offsets, final=False):
        cost = cost.toarray()
        if rowcol_offsets is not None:
            cost = cost + rowcol_offsets
        
        idx = lap_solvers.jv(cost, jv_backend=self.jv_backend)
        if final:
            return idx
