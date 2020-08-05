%%  Documentation
% This matlab script is made to generate a realistic data set
% for automotive radar interference, with multiple sources
% of interference ( ARIM-v2 ).
%
%%  Data set generator
% Variable interference slope/Signal slope limits are [0, 1.5]
% SIR limits for an interference source are [-5 , SNR + 5] dB

clear all;
% Setting a random generator for data set reproductibility
rnd_seed = 706 + nr_interferences;
rng(rnd_seed);

nr_interferences = 2;


snr_limits = [5, 40];
nr_samples = 6000;

sb0_mat = zeros(1, 1024);
sb_mat = zeros(1, 1024);
amplitude_mat = zeros(1, 2048);
distance_mat = zeros(1, 2048);
info_mat = zeros(1, 2*nr_interferences + 2);

index = 1;
for snr = snr_limits(1):5:snr_limits(2)
    for i=1:1:nr_samples
        % Variable interference slope/Signal slope
        slope_coef = randi([1,150],[1, nr_interferences]) / 100;
        sir_coef = randi([-5, snr + 5],[1, nr_interferences]);

        nr_targets = randi([1,4], 1);
        A = randi([1,100],1, nr_targets) / 100;
        A(randi([1,nr_targets])) = 1;
        teta = unifrnd(-pi,pi, 1, nr_targets);
        complexA = A.*exp(1i*teta);
        r = randi([2,95], 1, nr_targets);

        [sb0, sb, label, distance_label] = gen_signal(snr, sir_coef, slope_coef, complexA, r);

        sb0_mat(index, :) = sb0;
        sb_mat(index, :) = sb;
        amplitude_mat(index, :) = label;
        distance_mat(index, :) = distance_label;

        % Adding information about signal
        info_mat(index, :) = [nr_interferences, snr, sir_coef, slope_coef];
        
        index = index + 1;

    end
end

dataset_name = strcat('arim', int2str(nr_interferences), '.mat');
save(dataset_name, 'sb0_mat' , 'sb_mat', 'amplitude_mat', 'distance_mat', 'info_mat');



