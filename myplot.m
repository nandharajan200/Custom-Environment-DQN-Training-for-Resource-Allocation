% Plotting Script

% Load the saved data
load('dqn_training_data.mat', 'eGreedySucCnt');

MaxTime = size(eGreedySucCnt, 2);
Ndev = size(eGreedySucCnt, 1);

% Drawing the results -----------------------------------

t = 1:MaxTime; % Time vector

% Plotting eGreedySucCnt
figure
hold on
for i = 1:Ndev
    plot(t, eGreedySucCnt(i, :), 'DisplayName', sprintf('Device %d', i))
end
ylabel('Average Active success Rate')
xlabel('Transmission')
title('\epsilon-greedy algorithm - DQN')
legend('show')
hold off

% Correct the waterfall plot for DQN
figure
x = 1:MaxTime; % Episodes
y = 1:Ndev; % Devices
[X, Y] = meshgrid(x, y);
z = eGreedySucCnt;

c = gradient(z);
w = waterfall(X, Y, z, c);
xlabel('Transmission')
ylabel('Users')
zlabel('Success Rate')
title('\epsilon-greedy algorithm - DQN')
