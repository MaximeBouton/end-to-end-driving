% make super fancy plots for poster

clear
close all


%% Regression: prelu 3 3 3 .60


% load data

path = './prelu_3_3_3_60/';
suffix = '45';

train = load([path, 'train', suffix, '.dat']);
val   = load([path, 'val',   suffix, '.dat']);
test  = load([path, 'test',  suffix, '.dat']); 

epoches = (1:numel(train))';


% make plot

f = figure(1);
set(f, 'Position', [200, 200, 800, 600]);
plot(epoches, train, epoches, val, epoches, test);
l = legend('\textbf{training loss}', '\textbf{validation loss}', '\textbf{test loss}');
set(l, 'Location', 'best', 'Interpreter', 'LaTex');
xlabel('\textbf{epoch}', 'Interpreter', 'LaTex');
ylabel('\textbf{loss}', 'Interpreter', 'LaTex');
xlim([1, numel(epoches)])
% ylim([0,2])
% yticks(0:.5:2)
set(gca,'TickLabelInterpreter','LaTex')
grid()
plotfixer


%% Regression: prelu 3 2 3 .60

% load data

path = './prelu_3_2_3_60/';
suffix = '_loss_prelu_3_2_3_6';

train = [load([path, 'train', suffix, '.dat']); ...
    load([path, 'train', suffix, '_cntd.dat'])];
val = [load([path, 'val', suffix, '.dat']); ...
    load([path, 'val', suffix, '_cntd.dat'])];
test = [load([path, 'test', suffix, '.dat']); ...
    load([path, 'test', suffix, '_cntd.dat'])];
% val   = load([path, 'val',   suffix, '.dat']);
% test  = load([path, 'test',  suffix, '.dat']); 
epoches = (1:numel(train))';


% make plot


f = figure(2);
set(f, 'Position', [200, 200, 800, 600]);
plot(epoches, train, epoches, val, epoches, test);
l = legend('\textbf{training loss}', '\textbf{validation loss}', '\textbf{test loss}');
set(l, 'Location', 'best', 'Interpreter', 'LaTex');
xlabel('\textbf{epoch}', 'Interpreter', 'LaTex');
ylabel('\textbf{loss}', 'Interpreter', 'LaTex');
xlim([1, numel(epoches)])
% ylim([0,2])
% yticks(0:.5:2)
set(gca,'TickLabelInterpreter','LaTex')
grid()
plotfixer

%% Classification: loss

% close all

% load data

path = './classification/';
suffix = '_loss';

train = load([path, 'train', suffix, '.dat']);
val   = load([path, 'val',   suffix, '.dat']);
test  = load([path, 'test',  suffix, '.dat']); 

epoches = (1:numel(train))';


% make plot

f = figure(3);
set(f, 'Position', [200, 200, 800, 600]);
plot(epoches, train, epoches, val, epoches, test);
l = legend('\textbf{training loss}', '\textbf{validation loss}', '\textbf{test loss}');
set(l, 'Location', 'best', 'Interpreter', 'LaTex');
xlabel('\textbf{epoch}', 'Interpreter', 'LaTex');
ylabel('\textbf{loss}', 'Interpreter', 'LaTex');
xlim([1, numel(epoches)])
ylim([0,2])
yticks(0:.5:2)
set(gca,'TickLabelInterpreter','LaTex')
grid()
plotfixer



%% Classification: accuracy

% close all

% load data

path = './classification/';
suffix = '_acc';

train = load([path, 'train', suffix, '.dat']);
val   = load([path, 'val',   suffix, '.dat']);
test  = load([path, 'test',  suffix, '.dat']); 

epoches = (1:numel(train))';


% make plot

f = figure(4);
set(f, 'Position', [200, 200, 800, 600]);
plot(epoches, train, epoches, val, epoches, test);
l = legend('\textbf{training accuracy}', '\textbf{validation accuracy}', '\textbf{test accuracy}');
set(l, 'Location', 'best', 'Interpreter', 'LaTex');
xlabel('\textbf{epoch}', 'Interpreter', 'LaTex');
ylabel('\textbf{accuracy}', 'Interpreter', 'LaTex');
xlim([1, numel(epoches)])
% ylim([0,2])
% yticks(0:.5:2)
set(gca,'TickLabelInterpreter','LaTex')
grid()
plotfixer




