%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  SELECT A NICE TEST RANGE 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% IMPORT DATA 

A = csvread('../../../data/regressionSteering2.csv',1);

steering = A(:,2);
steering = (steering-mean(steering))/std(steering);
y = [steering(57820:59830);steering(63470:68050)];

%% Plot data

plot(y)
