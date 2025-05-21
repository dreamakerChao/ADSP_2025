import numpy as np
import matplotlib.pyplot as plt

class MiniMaxFIRFilter:
    def __init__(self, N, fs, delta,cutoff,f1,f2):
        self.N = N
        self.fs = fs
        self.delta = delta
        self.cutoff = cutoff
        self.f1 = f1
        self.f2 = f2
        self.k = (N - 1) // 2
        #self.Fn =np.linspace(0, 0.5, self.k+2)
        self.Fn = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.iteration = 0
        self.E1 = 1000

    def _construct_matrix_A(self):
        """Construct matrix A and desired response hd."""
        A = np.zeros([self.k + 2, self.k + 2])
        hd = np.zeros(self.k + 2,np.int32)
    
        cutoff = self.cutoff / self.fs
        
        for m in range(self.k + 2):
            A[m, :self.k + 1] = np.cos(2 * np.pi * np.arange(self.k + 1) * self.Fn[m])
            
            # Set the last column
            A[m, self.k + 1] = (-1) ** m * 1 / (1 if self.Fn[m] > cutoff else 0.6)
            
            # Set the desired response hd[m] based on the cutoff condition
            hd[m] = int(self.Fn[m] >= cutoff)
            
        return A, hd

    def _calculate_error(self, R, hd, w):
        """Calculate the weighted error."""
        return (R - hd) * w

    def _find_extrema(self, err, F):
        """Find the extrema points and update Fn."""
        extreme_point, extreme_value = [], []
        boundary_point, boundary_value = [], []
        n = F.shape[0]

        for i in range(n):
            curr_err = err[i]
            if i == 0:  # First point
                if curr_err > 0 and curr_err > err[i + 1]:
                    boundary_point.append(F[i])
                    boundary_value.append(abs(curr_err))
                elif curr_err < 0 and curr_err <= err[i + 1]:
                    boundary_point.append(F[i])
                    boundary_value.append(abs(curr_err))
            elif i == n - 1:  # Last point
                if curr_err > err[i - 1] and curr_err > 0:
                    boundary_point.append(F[i])
                    boundary_value.append(abs(curr_err))
                elif curr_err < err[i - 1] and curr_err < 0:
                    boundary_point.append(F[i])
                    boundary_value.append(abs(curr_err))
            else:  # Middle points
                prev_err = err[i - 1]
                next_err = err[i + 1]
                if curr_err > prev_err and curr_err > next_err:
                    extreme_point.append(F[i])
                    extreme_value.append(curr_err)
                elif curr_err < prev_err and curr_err < next_err:
                    extreme_point.append(F[i])
                    extreme_value.append(curr_err)

        # Step 2: If we have fewer than k+2 extreme points, select the smallest error points
        while len(extreme_value) < (self.k + 2):
            if len(boundary_value) > 0:
                # Find the point with the smallest error value (minimizing error)
                min_err_index = np.argmin(boundary_value)
                min_err_value = boundary_value.pop(min_err_index)
                min_err_point = boundary_point.pop(min_err_index)
                
                extreme_value.append(min_err_value)
                extreme_point.append(min_err_point)
            else:
                break

        # Step 3: Return the sorted extreme points and their corresponding values
        return sorted(zip(extreme_point, extreme_value))

    def _calculate_frequency_response(self, s, F):
        """Calculate the frequency response R[f]."""
        R = np.zeros(F.shape[0])
        for f in range(F.shape[0]):
            temp = sum(s[n] * np.cos(2 * np.pi * n * F[f]) for n in range(self.k + 1))
            R[f] = temp
        return R

    def _update_F0(self, extreme_point):
        """Update F0 from the extreme points."""
        self.Fn = np.array([ep[0] for ep in extreme_point])

    def _plot_error(self, F, err):
        """Plot the error for the current iteration."""
        max_err = np.max(np.abs(err))
        max_err_idx = np.argmax(np.abs(err))
        plt.figure()
        plt.plot(F, err)
        plt.scatter(F[max_err_idx], err[max_err_idx], c='r', marker='x', s=80, label=f'Max Error = {max_err:.5f}')
        plt.legend()
        plt.xlabel('Frequency')
        plt.ylabel('Error')
        plt.title(f'Iteration {self.iteration}')
        plt.show()
        return max_err
    
    def _plot_impulse(self,s):
        # impulse response
        h_n = np.zeros([self.N])
        y = np.zeros([self.N])

        for i in range(0, self.N):
            if i < self.k:
                h_n[i] = s[self.k-i] / 2
            elif i == self.k:
                h_n[i] = s[0]
            elif i > self.k:
                h_n[i] = s[i-self.k] / 2

        n = np.arange(self.N)
        plt.figure(figsize=(10, 6))
        plt.stem(n, h_n)

        plt.plot(n, y, color='b', linestyle='-', label='Zero Reference')
        plt.xlabel('n', fontsize=14)
        plt.ylabel('h[n]', fontsize=14)
        plt.title('Impulse Response', fontsize=16)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _plot_response(self,F,R,hd,extreme_point):
        extreme_R = [R[F == f][0] for f in extreme_point]

        plt.figure(figsize=(8, 6))
        plt.plot(F, R, label='H(F)')
        plt.plot(F, hd, label='Hd(F)')
        plt.scatter(extreme_point, extreme_R, color='r', marker='o', label='Extremes')

        plt.axvline(x=self.f1/self.fs, color='g', linestyle='--', label='Transition Start')
        plt.axvline(x=self.f2/self.fs, color='r', linestyle='--', label='Transition End')

        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.xlim(0, 0.5)
        plt.ylim(-0.2, 1.2)
        plt.legend()
        plt.title('Frequency Response')
        plt.grid(True)
        plt.show()

    ## design
    def design_filter(self):
        """Design the Mini-Max FIR filter."""
        while True:
            # Step 2: Construct matrix A and desired response hd
            A, hd = self._construct_matrix_A()

            # Solve for s
            s = np.linalg.inv(A).dot(hd)

            # Step 3: Calculate frequency response R[f] and error
            F = np.linspace(0, 0.5, 5001)
            R = self._calculate_frequency_response(s, F)

            # Calculate the desired response hd and weighting function w
            hd = np.array([int(f >= self.cutoff / self.fs) for f in F])
            w = np.array([0.6 if f <= self.f1 / self.fs else 1 if f >= self.f2 / self.fs else 0 for f in F])

            # Step 3: Calculate error
            err = self._calculate_error(R, hd, w)

            # Step 4: Find extrema and update F0
            extreme_point = self._find_extrema(err, F)
            self._update_F0(extreme_point)

            # Plot the error for this iteration
            max_err = self._plot_error(F, err)

            

            # Step 5: Check for convergence
            E0 = max(abs(err))
            max_err = np.max(np.abs(err))
            if self.iteration >= 0:
                print(f'Iteration {self.iteration} max_err is : {max_err:.5f}')

            if abs(self.E1 - E0) < self.delta:
                break
            else:
                self.E1 = E0
                self.iteration += 1

        extreme_point = np.array(extreme_point)[:,0]
        max_err = self._plot_error(F, err)
        self._plot_impulse(s)
        self._plot_response(F,R,hd,extreme_point)

        return s
    ## design

if __name__ == "__main__":
    filter_design = MiniMaxFIRFilter(N=19, fs=8000, delta=0.0001,cutoff=2200,f1=2000,f2=2400)
    filter_coefficients = filter_design.design_filter()
    print(filter_coefficients)
    