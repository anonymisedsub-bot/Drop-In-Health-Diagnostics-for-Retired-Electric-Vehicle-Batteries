% =========================================================================
% Description: 
%   This script performs comprehensive preprocessing on battery cycling data
%   ('CETL-data.mat'). It includes anomaly filtering, multi-stage data 
%   alignment, monotonic interpolation, and Incremental Capacity (IC) 
%   curve computation. 
% =========================================================================

clear; clc; close all;

%% 1. Data Initialization
fprintf('Loading dataset...\n');
load('CETL-data.mat', 'unbalanced_data');
data = unbalanced_data;

% Initialize containers for processed data
processed_Vd = data.Vd;  
processed_Qd = data.Qd;
Qlabel = data.Capacity_labels;

%% 2. Quality Control & Anomaly Filtering
fprintf('Applying quality control filters...\n');

remove_idx1 = [];  

% Filter out sequences based on length and minimum voltage
for i = 1:length(data.Vp)
    Vdis_min = min(data.Vd{i});
    if (length(data.Qd{i}) < 1000) || (Vdis_min > 3000)  
        remove_idx1 = [remove_idx1; i];  
        continue
    else 
        Vd_data(i,:) = processed_Vd{i}(1:1000);
        Qd_data(i,:) = processed_Qd{i}(1:1000);
    end    
end 

% Filter outliers at specific boundaries
remove_idx2 = find(Qd_data(:,1000) > 10000);
remove_idx3 = find(Vd_data(:,1) < 3300);
remove_idx = unique([remove_idx1; remove_idx2; remove_idx3]);

% Apply deletions
processed_Vd(remove_idx) = [];  
processed_Qd(remove_idx) = [];  
Qlabel(remove_idx) = [];

%% 3. Stage-1 Coarse Interpolation (Voltage Range Alignment)
fprintf('Executing Stage-1 coarse interpolation...\n');

voltage_range = 3300:-3:3000; 
interpolated_Qd = zeros(length(processed_Vd), length(voltage_range));
windowLength = 50;
smooth_mode = '1';

for i = 1:length(processed_Vd)
    % Apply smoothing based on predefined mode
    switch smooth_mode
        case '1'     
            Vd_unique = smooth(processed_Vd{i}, 'moving', 100);
            Qd_unique = smooth(processed_Qd{i}, 'moving', 100);  
        case '0'
            Vd_unique = processed_Vd{i};
            Qd_unique = processed_Qd{i}; 
    end

    % Locate the first index k where V <= 2600 and crop subsequent data
    k = find(Vd_unique <= 2600, 1, 'first');
    if ~isempty(k)
        Vd_unique = Vd_unique(1:k);  
        Qd_unique = Qd_unique(1:k);  
    end

    % Sliding window interpolation
    for j = 1:length(Vd_unique)
        if j < windowLength
            Vd_window = Vd_unique(1:j);
            Qd_window = Qd_unique(1:j);
        else 
            Vd_window = Vd_unique(j-windowLength+1:j);
            Qd_window = Qd_unique(j-windowLength+1:j);
        end

        [Vd_window, idx_window] = unique(Vd_window);
        Qd_window = Qd_window(idx_window);

        idxVdInWindow = find(voltage_range > min(Vd_window) & voltage_range < max(Vd_window));
        QdInWindow = interpolated_Qd(i, idxVdInWindow);
        idxQdInWindow = find(QdInWindow == 0);
        idxInWindow = idxVdInWindow(idxQdInWindow);

        if ~isempty(idxInWindow)              
            interpolated_Qd(i, idxInWindow) = interp1(Vd_window, Qd_window, voltage_range(idxInWindow), 'linear', 'extrap'); 
        end
    end
end

%% 4. Stage-2 High-Resolution Interpolation & IC Extraction
fprintf('Executing Stage-2 high-resolution interpolation and IC calculation...\n');

% Core Parameters
Vtr = 2.8;                       % Retain tail section where V < Vtr
Vinterval = 5e-3;
Vd_grid = 2.5 : Vinterval : 3.5; % Target voltage grid
Ngrid = numel(Vd_grid);
mono_tol = 1e-6;                 % Monotonicity tolerance (Volt magnitude)

% Preallocation
processed_Vd_mat = nan(Ngrid, numel(processed_Vd)); 
processed_Qd_mat = nan(Ngrid, numel(processed_Vd)); 

for i = 1:numel(processed_Vd)
    
    % 4.1 Local smoothing and unit conversion
    Vd = processed_Vd{i} / 1000;                     % mV -> V
    Vd = smoothdata(Vd, 'sgolay', 20);               
    Qd = processed_Qd{i} / 1000;                     % mAh -> Ah

    % Retain tail section (V < Vtr) and eliminate rebound
    idx_Vd = find(Vd < Vtr);
    if isempty(idx_Vd), continue; end
    
    Vtail = Vd(idx_Vd);
    [~, idx_tail] = min(Vtail);
    idx_last = idx_Vd(idx_tail);
    Vd = Vd(1:idx_last);
    Qd = Qd(1:idx_last);

    % Reverse consistency (ensure monotonic direction)
    Vd = Vd(end:-1:1);
    Qd = Qd(end:-1:1);

    % 4.2 Segment Vd into monotonic runs
    segs = split_into_monotone_runs(Vd, mono_tol);
    if numel(segs) > 100
        continue; % Skip sample if excessively noisy
    end

    % 4.3 PCHIP Interpolation on target grid
    Qd_interp = nan(Ngrid, 1);

    % Inter-segment interpolation
    for k = 1:numel(segs)
        s = segs{k}(1); e = segs{k}(2);
        v = Vd(s:e);    q = Qd(s:e);

        % Validity and uniqueness check
        mask_valid = ~isnan(v) & ~isnan(q);
        v = v(mask_valid); q = q(mask_valid);
        if numel(v) < 2, continue; end
        
        [vuniq, ia] = unique(v, 'stable'); quniq = q(ia);
        [vsort, ord] = sort(vuniq, 'ascend'); qsort = quniq(ord);
        if numel(vsort) < 2, continue; end

        inrange = (Vd_grid >= vsort(1)) & (Vd_grid <= vsort(end));
        if any(inrange)
            Qd_interp(inrange) = interp1(vsort, qsort, Vd_grid(inrange), 'pchip');
        end
    end

    % Boundary extrapolation
    mask_all = ~isnan(Vd) & ~isnan(Qd);
    Vall = Vd(mask_all); Qall = Qd(mask_all);
    
    if numel(Vall) >= 2
        [Vall_u, ia] = unique(Vall, 'stable'); Qall_u = Qall(ia);
        [Vall_s, ord] = sort(Vall_u, 'ascend'); Qall_s = Qall_u(ord);

        low_mask  = isnan(Qd_interp) & (Vd_grid < Vall_s(1));
        high_mask = isnan(Qd_interp) & (Vd_grid > Vall_s(end));
        
        if any(low_mask)
            Qd_interp(low_mask) = interp1(Vall_s, Qall_s, Vd_grid(low_mask), 'linear', 'extrap');
        end
        if any(high_mask)
            Qd_interp(high_mask) = interp1(Vall_s, Qall_s, Vd_grid(high_mask), 'linear', 'extrap');
        end
    end

    % Nearest-neighbor fallback for internal NaNs
    nan_mask = isnan(Qd_interp);
    if any(nan_mask) && numel(Vall) >= 2
        Qd_interp(nan_mask) = interp1(Vall_s, Qall_s, Vd_grid(nan_mask), 'nearest', 'extrap');
    end
    
    Qd_interp = Qd_interp(end:-1:1);
    
    % Store in matrices
    processed_Vd_mat(:, i) = Vd_grid(:);
    processed_Qd_mat(:, i) = Qd_interp(:);
end

% Remove entirely NaN columns
mask_del = all(isnan(processed_Vd_mat), 1) | all(isnan(processed_Qd_mat), 1);
processed_Vd_mat(:, mask_del) = [];
processed_Qd_mat(:, mask_del) = [];

% 4.4 Calculate Incremental Capacity (IC / dQ/dV)
processed_qIC_mat = diff(processed_Qd_mat, 1) / Vinterval;
processed_qIC_mat = processed_qIC_mat(end:-1:1, :);
processed_qIC_mat = smoothdata(processed_qIC_mat, 'movmean', 5);
processed_vIC_mat = processed_Vd_mat(2:end, :);

fprintf('Processing complete.\n');

%% 5. Visualization (Nature-Style formatting)
% Note: Defaulted to standard coordinate axes.
font_name = 'Arial';
font_size = 12;

% Figure 1: Stage-1 Coarse Interpolation Output
figure('Color','w'); 
plot(interpolated_Qd, voltage_range, 'LineWidth', 1.2);
xlabel('Discharge capacity (mAh)', 'FontName', font_name, 'FontSize', font_size);
ylabel('Voltage (V)', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1.2, 'Box', 'on');

% Figure 2: Q(V) Interpolation Results
figure('Color','w'); 
plot(processed_Qd_mat(:,1:100:end), processed_Vd_mat(:,1:100:end), 'LineWidth', 1.2);
xlabel('Q_d (Ah)', 'FontName', font_name, 'FontSize', font_size); 
ylabel('V (V)', 'FontName', font_name, 'FontSize', font_size); 
title('Q(V) Interpolation Profiles', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1.2, 'Box', 'on');

% Figure 3: IC Curves
figure('Color','w'); 
plot(processed_vIC_mat(:,1:100:end), processed_qIC_mat(:,1:100:end), 'LineWidth', 1.2);
xlabel('Voltage (V)', 'FontName', font_name, 'FontSize', font_size); 
ylabel('IC (dQ/dV)', 'FontName', font_name, 'FontSize', font_size); 
title('IC Curves', 'FontName', font_name, 'FontSize', font_size);
set(gca, 'FontName', font_name, 'FontSize', font_size, 'LineWidth', 1.2, 'Box', 'on');

% Clean workspace, keeping only vital matrices
clearvars -except processed_Qd_mat processed_Vd_mat processed_qIC_mat processed_vIC_mat

fprintf('Saving processed dataset to CETL_data_processed.mat...\n');
save('CETL_data_processed.mat', 'processed_Qd_mat', 'processed_Vd_mat', 'processed_qIC_mat', 'processed_vIC_mat');

%% ==================== Auxiliary Functions ====================
function segs = split_into_monotone_runs(v, tol)
    % SPLIT_INTO_MONOTONE_RUNS Divides a vector into weakly monotonic segments.
    % Accounts for minor fluctuations using a tolerance (tol).
    
    v = v(:);
    n = numel(v);
    if n < 2
        segs = { [1, n] };
        return;
    end

    d = diff(v);
    dir = zeros(size(d));
    dir(d >  tol) =  1;
    dir(d < -tol) = -1;
    
    for k = 1:numel(dir)
        if dir(k) == 0
            if k == 1
                nz = find(dir ~= 0, 1, 'first');
                if ~isempty(nz)
                    dir(k) = dir(nz);
                else
                    dir(k) = 0; 
                end
            else
                dir(k) = dir(k-1);
            end
        end
    end

    if all(dir == 0)
        segs = { [1, n] };
        return;
    end

    change = [true; diff(dir(:)) ~= 0];   
    starts = find(change);
    ends   = [starts(2:end)-1; n];

    keep = (ends - starts + 1) >= 2;
    starts = starts(keep);
    ends   = ends(keep);

    segs = cell(numel(starts),1);
    for i = 1:numel(starts)
        segs{i} = [starts(i), ends(i)];
    end
end