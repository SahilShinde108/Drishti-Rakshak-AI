function results = Run_District_Simulation(config)
% Run_District_Simulation - Pure MATLAB discrete-event simulation fallback
% Simulates queueing and transmission delays for telemedicine network

    if nargin < 1
        config = struct('num_phcs', 50, 'num_doctors', 5, 'days', 365, 'has_edge_triage', true);
    end
    
    results = struct('throughput', 0, 'wait_times', [], 'utilization', 0, 'capacity_estimate', 0);
    
    % Simulation parameters
    total_minutes = config.days * 24 * 60;
    arrivals_per_day = 10; % per PHC
    total_expected_arrivals = config.num_phcs * arrivals_per_day * config.days;
    
    % Generate arrival times (Poisson process approximation)
    % Using exponential inter-arrival times
    lambda = (config.num_phcs * arrivals_per_day) / (24 * 60); % arrivals per minute
    
    inter_arrivals = exprnd(1/lambda, [round(total_expected_arrivals * 1.2), 1]);
    arrival_times = cumsum(inter_arrivals);
    arrival_times = arrival_times(arrival_times <= total_minutes);
    num_patients = length(arrival_times);
    
    % Triage filter (60% normal cases filtered if enabled)
    if config.has_edge_triage
        % Assume 60% of cases are normal
        is_referred = rand(num_patients, 1) > 0.60;
        arrival_times = arrival_times(is_referred);
        num_patients = length(arrival_times);
    end
    
    % Transmission delay (assume 4G average: 2 seconds ~ 0.03 mins)
    tx_delays = normrnd(0.03, 0.01, [num_patients, 1]);
    tx_delays = max(0.01, tx_delays);
    
    hospital_arrival_times = arrival_times + tx_delays;
    
    % Doctor service times (average 5 minutes)
    service_times = exprnd(5, [num_patients, 1]);
    
    % Queue simulation (multi-server queue)
    doctor_available_time = zeros(config.num_doctors, 1);
    wait_times = zeros(num_patients, 1);
    
    for i = 1:num_patients
        arr = hospital_arrival_times(i);
        
        % Find earliest available doctor
        [earliest_time, doc_idx] = min(doctor_available_time);
        
        if earliest_time > arr
            % Patient must wait
            wait = earliest_time - arr;
            wait_times(i) = wait;
            doctor_available_time(doc_idx) = earliest_time + service_times(i);
        else
            % Doctor is free
            wait_times(i) = 0;
            doctor_available_time(doc_idx) = arr + service_times(i);
        end
    end
    
    total_busy_time = sum(service_times);
    total_doctor_time = total_minutes * config.num_doctors;
    
    results.throughput = num_patients / config.days;
    results.wait_times = wait_times;
    results.mean_wait_time = mean(wait_times);
    results.utilization = (total_busy_time / total_doctor_time) * 100;
    results.capacity_estimate = num_patients;
    
    disp('Simulation Complete.');
    fprintf('Throughput: %.2f patients/day\n', results.throughput);
    fprintf('Mean Wait Time: %.2f minutes\n', results.mean_wait_time);
    fprintf('Doctor Utilization: %.2f%%\n', results.utilization);
    
end
