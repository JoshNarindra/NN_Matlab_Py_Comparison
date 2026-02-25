
%% Load Data
data = readtable('Dry_Bean_Dataset.xlsx');

%% Inspect Data
size(data);
head(data);
summary(data);

%% Seperate Features and Labels
X = data{:,1:end-1}; % Data Features for Every Observation. Exclude last column (labels).
Y = data{:,end}; % Data labels from the last column for every observation.

%% Encode Labels into Categorical Integer Format
Y = categorical(Y); % Encode Data from Class names (labels) to Categorical Integers mapped to Label Names.

%head(Y) %Check
%% Split the Training and Testing Data 80/20 (Stratified)
cv = cvpartition(Y, 'HoldOut', 0.2);

XTrain = X(training(cv),:); % Training data features
YTrain = Y(training(cv)); % Corresponding labels for the training data
XTest = X(test(cv),:);    % Test data features
YTest = Y(test(cv));      % Corresponding labels for the test data

%% Feature Scaling

mu = mean(XTrain);
sigma = std(XTrain);

XTrain = (XTrain - mu) ./ sigma; % Standardize training features
XTest = (XTest - mu) ./ sigma;   % Standardize test features using training mean and std

%% Format Data for Model Building

XTrain = XTrain';
XTest = XTest';

YTrain_dummy = dummyvar(YTrain)';
YTest_dummy = dummyvar(YTest)';

%head(YTest_dummy);

%% Build Single Hidden Layer Multilayer Perceptron

% Allow for varying of neurons in hidden layer
hiddenLayerSize = 32;
net = patternnet(hiddenLayerSize);

%% Train Network

%net = train(net,)

%% Predictions

%% Evaluate Performance

%% Tune Hidden Size and Learning Rate
