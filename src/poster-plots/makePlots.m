% make super fancy plots for poster

clear
close all


%% Regression: prelu 3 3 3 .60

close all


% load data

path = './prelu_3_3_3_60/';
suffix = '45';

train = load([path, 'train', suffix, '.dat']);
val   = load([path, 'val',   suffix, '.dat']);
test  = load([path, 'test',  suffix, '.dat']); 

epoches = (1:numel(train))';


% make plot

f = figure(1);
set(f, 'Position', [50, 50, 800, 600]);
plot(epoches, train, epoches, val, epoches, test)
l = legend('training loss', 'validation loss', 'test loss');
set(l, 'Location', 'best');
xlabel('epoche', 'Interpreter', 'LaTex');
ylabel('loss', 'Interpreter', 'LaTex');
plotfixer


%% Regression: prelu 3 2 3 .60

% load data

suffix = '45';

train = load('train' + suffix + '.dat');
val   = load('train' + suffix + '.dat');
test  = load('train' + suffix + '.dat');

epoches = (1:numel(train))';


% make plot

f = figure(1);
set(f, 'Position', [50, 50, 800, 600]);
plot(epoches, train, epoches, val, epoches, test)
l = legend('training loss', 'validation loss', 'test loss');
set(l, 'Location', 'best');
xlabel('epoche', 'Interpreter', 'LaTex');
plotfixer


%% Classification: loss

close all

% load data

path = './classification/';
suffix = '_loss';

train = load([path, 'train', suffix, '.dat']);
val   = load([path, 'val',   suffix, '.dat']);
test  = load([path, 'test',  suffix, '.dat']); 

epoches = (1:numel(train))';


% make plot

f = figure(1);
set(f, 'Position', [50, 50, 1000, 600]);
plot(epoches, train, epoches, val, epoches, test);
l = legend('training loss', 'validation loss', 'test loss');
set(l, 'Location', 'best', 'Interpreter', 'LaTex');
xlabel('epoch', 'Interpreter', 'LaTex');
ylabel('loss', 'Interpreter', 'LaTex');
xlim([1, numel(epoches)])
ylim([0,2])
yt = [0,1,2];
ax = gca;
% set(gca,'TickLabelInterpreter','LaTex')
% set(gca, 'YTick', yt)
plotfixer





