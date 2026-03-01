
%% Load Data
data = readtable('Dry_Bean_Dataset.xlsx');

%% Inspect Data
% No Missing Data Dataset Pre Cleaned

size(data)
%head(data);
%summary(data);

%% Determine the number of Unique Classes - or Output Labels
numel(unique(data.Class))

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

%% Format Label Data for Model Building

YTrain_dummy = dummyvar(YTrain);
YTest_dummy = dummyvar(YTest);

%head(YTest_dummy);

%% Build Single Hidden Layer Multilayer Perceptron

% Allow for varying of neurons in hidden layer
hiddenLayer = 3;
inputFeatures = 16;
outputLayer = 7;

% Input Layer to Hidden Layer
W1 = randn(inputFeatures, hiddenLayer);
B1 = zeros(1, hiddenLayer);

Z1 = XTrain * W1 + B1;
A1 = sigmoid(Z1);

% Hidden Layer to Output Layer
W2 = randn(hiddenLayer,outputLayer);
B2 = zeros(1, outputLayer);

Z2 = A1 * W2 + B2;
A2 = sigmoid(Z2);

% Cross Entropy Loss
loss = cross_entropy(A2, YTrain_dummy);

% Mean Square Error Loss
mse_loss = mse_loss(YTrain_dummy, A2);

% Backpropogation 
[numSamples, numFeatures] = size(XTrain);



%% Train Network


%% Predictions

%% Evaluate Performance

%% Tune Hidden Size and Learning Rate


%% Functions

% Define Sigmoid Activation Function
function output = sigmoid(z)
    output = 1./ (1 + exp(-z));
end

% Cross Entropy Loss
function loss = cross_entropy(predictions, true_labels)
m = size(predictions, 1);% number of samples
% Add small epsilon to avoid log(0)
loss = -sum(sum(true_labels .* log(predictions + 1e-8))) / m;
end 

function output = sigmoid_derivative(a)
    % Derivative of sigmoid: a * (1 - a)
    output = a .* (1 - a);
end

% MSE Loss
function loss = mse_loss(y_true, y_pred)
    % Mean Squared Error loss (matching Python)
    loss = 0.5 * mean(sum((y_true - y_pred).^2, 2));
end