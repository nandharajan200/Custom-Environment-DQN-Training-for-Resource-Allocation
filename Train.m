% Main DQN Training and Saving Script

% Training parameters
Ndev = 8; % Number of devices
Nchs = 4; % Number of channels
Npwr = 2; % Number of power levels
MaxTime = 5; % Number of episodes

% Environment parameters
SNRdB = 10;
SNR2dB = 10;
SINR_TH1dB = -10;
SINR_TH2dB = -10;
SINR_TH_HdB = -10;
SINR_TH_LdB = -10;

sinr_th1 = 10^(SINR_TH1dB/10);
sinr_th2 = 10^(SINR_TH2dB/10);
sinr_thH = 10^(SINR_TH_HdB/10);
sinr_thL = 10^(SINR_TH_LdB/10);

PwH = 0.8;
PwL = 0.2;

Vkm = 1;
Wkm = 1;
LowR = 0;
CostR = 0;

% Initialize global variables
global eGreedySucCnt;
eGreedySucCnt = zeros(MaxTime, Ndev); % Rows for episodes, columns for devices

% Initialize parameters
gamma = 0.99;
alpha = 0.1;
epsilon = 0.9;
epsilon_decay = 0.99;
epsilon_min = 0.01;

% Define the neural network for DQN
obsInfo = rlNumericSpec([Ndev * Nchs, 1], 'LowerLimit', -inf, 'UpperLimit', inf);
actInfo = rlFiniteSetSpec(1:Nchs * Npwr);

dnn = [
    featureInputLayer(obsInfo.Dimension(1), 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(24, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    fullyConnectedLayer(length(actInfo.Elements), 'Name', 'fc3')
    regressionLayer('Name', 'output')];

% Create the Q-value function representation
qValueRepresentation = rlQValueRepresentation(dnn, obsInfo, actInfo, 'Observation', {'state'});

% Specify the agent options
agentOpts = rlDQNAgentOptions(...
    'UseDoubleDQN', true, ...
    'TargetSmoothFactor', 1e-3, ...
    'DiscountFactor', 0.99, ...
    'ExperienceBufferLength', 1e6, ...
    'MiniBatchSize', 64, ...
    'EpsilonGreedyExploration', rl.option.EpsilonGreedyExploration('Epsilon', epsilon, 'EpsilonDecay', epsilon_decay, 'EpsilonMin', epsilon_min));

% Create the DQN agent
agent = rlDQNAgent(qValueRepresentation, agentOpts);

% Training options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', MaxTime, ...
    'MaxStepsPerEpisode', 200, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', MaxTime);

% Create the custom environment
env = rlFunctionEnv(obsInfo, actInfo, @stepFunction, @resetFunction);

% Train the agent
trainingStats = train(agent, env, trainOpts);

% Save the success count data to a .mat file
save('dqn_training_data.mat', 'eGreedySucCnt', 'trainingStats');

%% Supporting Functions

function [nextObs, reward, isDone, loggedSignals] = stepFunction(action, loggedSignals)
    global Ndev Nchs Npwr sinr_thH sinr_thL PwH PwL SNRdB SNR2dB Vkm Wkm LowR CostR;
    global eGreedySucCnt;

    state = loggedSignals.State;
    reward = 0; % Default reward initialization

    % Example reward calculation
    reward = sum(state); % Example: reward as sum of state values
    if ~isfinite(reward)
        error('Reward is not a finite scalar.');
    end

    % Update success count for each device based on state and action
    currentEpisode = loggedSignals.Episode;
    if currentEpisode > 0 && currentEpisode <= size(eGreedySucCnt, 1)
        % Assuming 'action' is an index related to the device
        deviceIndex = mod(action - 1, Ndev) + 1; % Ensure deviceIndex is within 1 to Ndev

        % Define criteria for success based on state and action
        % Example: Success if the sum of state values exceeds a threshold
        successThreshold = 5; % Define a threshold based on your needs
        if sum(state) > successThreshold
            eGreedySucCnt(currentEpisode, deviceIndex) = eGreedySucCnt(currentEpisode, deviceIndex) + 1;
        end
    else
        error('Episode number out of bounds.');
    end

    % Determine if episode is done
    isDone = (sum(state) > 50); % Example condition for episode termination
    if isDone
        % Increment the episode count
        loggedSignals.Episode = loggedSignals.Episode + 1;
    end

    % Transition to next state
    nextObs = state; % For simplicity, keep the state same; update as needed
    loggedSignals.State = nextObs; % Update state in logged signals
end
function [initialState, loggedSignals] = resetFunction()
    global Ndev Nchs MaxTime;

    % Initialize state
    initialState = rand(Ndev * Nchs, 1);

    % Check if the episode count needs to be incremented
    if isempty(getappdata(0, 'currentEpisode'))
        % Initialize the episode count if it's the first reset
        setappdata(0, 'currentEpisode', 1);
    else
        % Otherwise, just retrieve the current episode count
        currentEpisode = getappdata(0, 'currentEpisode');
        
        if currentEpisode >= MaxTime
            % Reset to 1 if maximum episodes reached
            setappdata(0, 'currentEpisode', 1);
        else
            % Increment episode count
            setappdata(0, 'currentEpisode', currentEpisode + 1);
        end
    end

    % Retrieve the updated episode count
    currentEpisode = getappdata(0, 'currentEpisode');

    % Update logged signals
    loggedSignals.State = initialState;
    loggedSignals.Episode = currentEpisode; % Set the current episode in logged signals
end

