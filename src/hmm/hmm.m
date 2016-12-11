

% load true labels and predictions
p = load('test.dat');
t = load('labels.dat'); %truth
t = readtable('clean-test.csv');
t = angle1;

t = (t-mean(t))/std(t);


% show plot
T = (1:numel(t))';
figure(1)
plot(T, t, T, p)
l = legend('true', 'prediction');
set(l, 'Interpreter', 'LaTex')
xlabel('t', 'Interpreter', 'LaTex')
ylabel('steering command', 'Interpreter', 'LaTex')
set(gca,'TickLabelInterpreter','LaTex')
%plotfixer



close all

%% get the ranges

% ranges
%ranges = [1, 1591, 4713, 8412, 9808, 11409, 11760, 12201];
ranges = [1, 2011, numel(angle1)]
for i = 1:(numel(ranges)-1)
    
    fprintf('\t%5.0f\t%5.0f\t%5.0f\n', ranges(i), ranges(i+1)-1, ranges(i+1)-1-ranges(i))
    
end

%% section-wise parameters

for i = 1:(numel(ranges)-1)
    
    r = ranges(i):(ranges(i+1)-1);
    n = ranges(i+1)-ranges(i);
    
    A = zeros(n-1, 2);
    A(:,1) = t(ranges(i):ranges(i+1)-2);
    A(:,2) = p((ranges(i)+1):(ranges(i+1)-1));
    
    b = zeros(n-1, 1);
    b(:) = t((ranges(i)+1):ranges(i+1)-1);
    
    w = (A'*A) \ (A'*b);
    
    fprintf('section %1.0f: w_1 = %1.4e, w_2 = %1.4e\n', i, w(1), w(2))
    
end


%% global parameters with last time step

n = numel(T) - numel(ranges) + 1;
A = zeros(n, 2);
b = zeros(n, 1);

for i = 1:(numel(ranges)-1)
    
    id1 = ranges(i);
    id2 = ranges(i+1)-1;
    
    A((id1-(i-1)):(id2-(i-1)-1),1) = t(id1:(id2-1));
    A((id1-(i-1)):(id2-(i-1)-1),2) = p((id1+1):id2);
    
    b((id1-(i-1)):(id2-(i-1)-1)) = t((id1+1):id2);
    
end

w = (A'*A) \ (A'*b);

fprintf('   global: w_1 = %1.4e, w_2 = %1.4e\n', w(1), w(2))


%% make final prediction from global parameters with last time step

p_final = zeros(numel(T), 1);

for i = 1:(numel(ranges)-1)
    
    id1 = ranges(i);
    id2 = ranges(i+1)-1;
    
    p_final(id1) = p(id1);
    
    for j = (id1+1):id2
        
        p_final(j) = p_final(j-1) * w(1) + p(j) * w(2);
        
    end
    
end


% show plot
T = (1:numel(t))';
figure(1)
plot(T, t, T, p, T, p_final)
l = legend('true', 'cnn', 'final');
set(l, 'Interpreter', 'LaTex')
xlabel('t', 'Interpreter', 'LaTex')
ylabel('steering command', 'Interpreter', 'LaTex')
set(gca,'TickLabelInterpreter','LaTex')
%plotfixer

close all

p_first = p_final;

%% global parameters with last two time steps

n = numel(T) - 2*numel(ranges) + 2;
A = zeros(n, 3);
b = zeros(n, 1);

for i = 1:(numel(ranges)-1)
    
    id1 = ranges(i);
    id2 = ranges(i+1)-1;
    
    A((id1-2*(i-1)):(id2-2*(i-1)-2),1) = t(id1:(id2-2));
    A((id1-2*(i-1)):(id2-2*(i-1)-2),2) = t((id1+1):(id2-1));
    A((id1-2*(i-1)):(id2-2*(i-1)-2),3) = p((id1+2):id2);
    
    b((id1-2*(i-1)):(id2-2*(i-1)-2)) = t((id1+2):id2);
    
end

w = (A'*A) \ (A'*b);

fprintf('   global: w_1 = %1.4e, w_2 = %1.4e, w_3 = %1.4e\n', w(1), w(2), w(3))


%% make final prediction from global parameters

p_final = zeros(numel(T), 1);

for i = 1:(numel(ranges)-1)
    
    
    
    id1 = ranges(i);
    id2 = ranges(i+1)-1;
    
    A((id1-2*(i-1)):(id2-2*(i-1)-2),1) = t(id1:(id2-2));
    A((id1-2*(i-1)):(id2-2*(i-1)-2),2) = t((id1+1):(id2-1));
    A((id1-2*(i-1)):(id2-2*(i-1)-2),3) = p((id1+2):id2);
    
    b((id1-2*(i-1)):(id2-2*(i-1)-2)) = t((id1+2):id2);
    
    
    id1 = ranges(i);
    id2 = ranges(i+1)-1;
    
    p_final(id1:(id1+1)) = p(id1:(id1+1));
    
    for j = (id1+2):id2
        
        p_final(j) = p_final(j-2) * w(1) + p_final(j-1) * w(2) + p(j) * w(3);
        
    end
    
end

p_second = p_final;
% %% global parameters with last three time steps 
% 
% n = numel(T) - 3*numel(ranges) + 3;
% A = zeros(n, 4);
% b = zeros(n, 1);
% 
% for i = 1:(numel(ranges)-1)
%     
%     id1 = ranges(i);
%     id2 = ranges(i+1)-1;
%     
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),1) = t(id1:(id2-3));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),2) = t((id1+1):(id2-2));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),3) = p((id1+2):(id2-1));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),4) = p((id1+3):(id2));
%     
%     b((id1-3*(i-1)):(id2-3*(i-1)-3)) = t((id1+3):id2);
%     
% end
% 
% w = (A'*A) \ (A'*b);
% 
% fprintf('   global: w_1 = %1.4e, w_2 = %1.4e, w_3 = %1.4e\n,  w_4 = %1.4e\n', w(1), w(2), w(3),w(4))
% 
% %% make final prediction from global parameters
% 
% p_final = zeros(numel(T), 1);
% 
% for i = 1:(numel(ranges)-1)
%     
%     
%     
%     id1 = ranges(i);
%     id2 = ranges(i+1)-1;
%     
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),1) = t(id1:(id2-3));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),2) = t((id1+1):(id2-2));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),3) = p((id1+2):(id2-1));
%     A((id1-3*(i-1)):(id2-3*(i-1)-3),4) = p((id1+3):(id2));
%     
%     b((id1-3*(i-1)):(id2-3*(i-1)-3)) = t((id1+3):id2);
%     
%     
%     id1 = ranges(i);
%     id2 = ranges(i+1)-1;
%     
%     p_final(id1:(id1+1)) = p(id1:(id1+1));
%     
%     for j = (id1+2):id2
%         
%         p_final(j) = p_final(j-2) * w(1) + p_final(j-1) * w(2) + p(j) * w(3) + w(4)*randn(1);
%         
%     end
%     
% end
%% show plot
T = (1:numel(t))';
figure(1)
% plot(T, t, T, p, T, p_final)
plot(T, t, T, p, T, p_first, T, p_second, T, p_final)
% l = legend('true', 'cnn', 'final');
l = legend('true', 'cnn', 'p1', 'p2', 'p3');
set(l, 'Interpreter', 'LaTex')
xlabel('t', 'Interpreter', 'LaTex')
ylabel('steering command', 'Interpreter', 'LaTex')
set(gca,'TickLabelInterpreter','LaTex')
%plotfixer

%% Print out performance metric
fprintf('RMS of CNN predictions: %6.6f\n',rms(t-p))
fprintf('RMS of first order smoothing: %6.6f\n',rms(t-p_first))
fprintf('RMS of second order smoothing: %6.6f\n',rms(t-p_second))
% fprintf('RMS of second order smoothing: %6.6f\n',rms(t-p_final))



