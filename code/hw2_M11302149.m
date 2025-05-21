function hw2()
    % ——— Filter Parameters ———
    k = 8;                         % Half length
    filt_Resp_Reso = 1024;        % FFT resolution
    Fp = 0.45;                    % Passband edge
    Fs = 0.55;                    % Stopband start
    N = 2 * k + 1;                % Filter length

    % ——— Frequency Sampling Method ———
    [fs_filt, F, H_F, r1, r] = freq_samp(k, N, Fp, Fs);

    % ——— Plot Time-Domain Signals ———
    figure;
    subplot(3,1,1);
    stem(0:N-1, real(r1), 'filled'); 
    title('r1[n]: Raw Circular IFFT'); grid on;

    subplot(3,1,2);
    stem(-k:k, real(r), 'filled'); 
    title('r[n]: Center-Shifted Linear Response'); grid on;

    subplot(3,1,3);
    stem(0:N-1, real(r), 'filled'); 
    title('h[n]: Causal Filter Impulse Response'); grid on;

    % ——— Plot Frequency Response ———
    freq_Resp_pos = (0:filt_Resp_Reso-1) / filt_Resp_Reso;
    padded = [fs_filt, zeros(1, filt_Resp_Reso - N)];
    arr_shift = circshift(padded, -k);
    freq_Resp = fft(arr_shift);

    figure;
    hold on;

    f_full = linspace(0, 1, 4096); 
    
    Hd_full = Hd_func(f_full,Fp,Fs);
    plot(f_full, imag(Hd_full), 'k-', 'LineWidth', 1.5, 'DisplayName', 'Ideal response');


    scatter(F, imag(H_F), 36, 'k', 'o', 'LineWidth', 1.2, 'DisplayName', 'Sampled points');
    
    plot(freq_Resp_pos, imag(freq_Resp), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Design response');
    
    yl = ylim;
    plot([Fp Fs], [yl(1) yl(1)], 'r--', 'HandleVisibility','off');
    plot([Fp Fp], yl, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Transition Band');
    plot([Fs Fs], yl, 'r--', 'LineWidth', 1.5, 'HandleVisibility','off');
    
    xlabel('Normalized Frequency (Hz)');
    ylabel('Imaginary Part');
    title('Frequency Response');
    legend('Location', 'northwest');
    grid on;
    


end

function [fs_filt, F, H_F, r1, r] = freq_samp(k, N, Fp, Fs)
    % ——— Frequency Sampling Points ———
    F = (0:N-1) / N;
    f = F;
    f(f > 0.5) = f(f > 0.5) - 1;

    % ——— Desired Frequency Response Definition ———
    H_F = zeros(1, N);
    mask_p = abs(f) <= Fp;
    mask_s = abs(f) >= Fs;
    mask_t = ~mask_p & ~mask_s;

    % Passband: ideal differentiator
    H_F(mask_p) = 1j * 2 * pi * f(mask_p);
    % Stopband: zero
    H_F(mask_s) = 0;
    % Transition band: linear interpolation
    H_F(mask_t) = 1j * 2 * pi * Fp .* ...
                  (Fs - abs(f(mask_t))) ./ (Fs - Fp) .* ...
                  sign(f(mask_t));

    % ——— Circular IFFT to Get r1[n] ———
    r1 = ifft(H_F);

    % ——— Center Shift to Get Symmetric r[n] ———
    r = circshift(r1, [0, k]);

    % ——— Final Output: Causal Impulse Response ———
    fs_filt = r;
end

function y = Hd_func(x, Fp, Fs)
    % Hd_func   Piecewise ideal frequency response (imaginary only)
    %   y = Hd_func(x, Fp, Fs)
    %   Inputs:
    %       x  - normalized frequency values (0 <= x <= 1)
    %       Fp - passband edge frequency
    %       Fs - stopband start frequency
    %   Output:
    %       y  - complex-valued ideal response (imaginary part only)
    
        y1 = 2 * pi * Fp;
        y2 = -2 * pi * (1 - Fs);
    
        y = (x <= Fp) .* (1j * 2 * pi * x) + ...
            ((x > Fp) & (x < Fs)) .* (1j * ((y2 - y1)/(Fs - Fp) * (x - Fp) + y1)) + ...
            (x >= Fs) .* (-1j * 2 * pi * (1 - x));
    end
    