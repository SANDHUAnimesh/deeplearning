clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Data 

% create an image data store from the raw images 
imdsTrain = imageDatastore('speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

% create an image validation data store from the validation images 
imdsVal = imageDatastore('speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

%% figure;
perm = randperm(numel(imdsTrain.Files), 9);
for i = 1:9
    subplot(3,3,i);
    img = readimage(imdsTrain, perm(i));
    imshow(img);
    title(string(imdsTrain.Labels(perm(i))));
end
%% % Define augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-3 3], ... % Random shifts in X-direction
    'RandYTranslation', [-3 3], ... % Random shifts in Y-direction
    'RandXScale', [0.9 1.1], ... % Random scaling in X-direction
    'RandYScale', [0.9 1.1]); % Random scaling in Y-direction

% Apply augmentation to training data
imageSize = [98 50 1];
augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter);

% Keep validation data unchanged
augimdsVal = imdsVal;

%% % Define improved CNN architecture
layers = [
    imageInputLayer([98 50 1]) 

    convolution2dLayer(3, 32, 'Padding', 'same') % Increased filters (16 → 32)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same') % Increased filters (32 → 64)
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.4) % Increased dropout to prevent overfitting

    fullyConnectedLayer(12)
    softmaxLayer
    classificationLayer];

% Retrain the CNN with augmentation
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(augimdsTrain, layers, options);
%% % Define possible configurations
filterSizes = [32, 64]; % Number of filters in Conv layers
numConvLayers = [2, 3]; % Number of Conv blocks

% Store results
results = [];

for f = 1:length(filterSizes)
    for l = 1:length(numConvLayers)
        
        % Define CNN dynamically
        layers = [
            imageInputLayer([98 50 1])];

        for i = 1:numConvLayers(l) % Loop over conv layers
            layers = [layers;
                convolution2dLayer(3, filterSizes(f), 'Padding', 'same')
                batchNormalizationLayer
                reluLayer
                maxPooling2dLayer(2, 'Stride', 2)];
        end

        layers = [layers;
            dropoutLayer(0.4)
            fullyConnectedLayer(12)
            softmaxLayer
            classificationLayer];

        % Training options
        options = trainingOptions('adam', ...
            'MaxEpochs', 50, ...
            'MiniBatchSize', 32, ...
            'InitialLearnRate', 0.001, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', augimdsVal, ...
            'Verbose', false);

        % Train the model
        net = trainNetwork(augimdsTrain, layers, options);

        % Evaluate
        YPred = classify(net, imdsVal);
        YValidation = imdsVal.Labels;
        accuracy = sum(YPred == YValidation) / numel(YValidation);

        % Store results
        results = [results; filterSizes(f), numConvLayers(l), accuracy*100];
        fprintf('Filters: %d, Layers: %d, Accuracy: %.2f%%\n', filterSizes(f), numConvLayers(l), accuracy*100);
    end
end

% Display results table
disp(array2table(results, 'VariableNames', {'Filters', 'Layers', 'Accuracy'}));
%% % Number of ensemble models
numModels = 3;

% Store trained networks
nets = cell(1, numModels);

% Define the best CNN architecture (32 filters, 3 layers)
layers = [
    imageInputLayer([98 50 1])

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)

    dropoutLayer(0.4)

    fullyConnectedLayer(12)
    softmaxLayer
    classificationLayer];

% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.001, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'Verbose', false);

% Train 3 CNN models with different random subsets
for i = 1:numModels
    fprintf('Training Model %d/%d...\n', i, numModels);
    
    % Create a new training subset (random sampling)
    subsetIdx = randperm(numel(imdsTrain.Files), round(0.8 * numel(imdsTrain.Files)));
    imdsSubset = subset(imdsTrain, subsetIdx);
    
    % Train the model
    nets{i} = trainNetwork(imdsSubset, layers, options);
end

fprintf('All %d models trained successfully!\n', numModels);
%% % Get predictions from all models
YPred1 = classify(nets{1}, imdsVal);
YPred2 = classify(nets{2}, imdsVal);
YPred3 = classify(nets{3}, imdsVal);

% Convert categorical labels to arrays
YPred = [YPred1, YPred2, YPred3];

% Majority voting (mode of predictions)
finalPredictions = mode(YPred, 2);

% Get actual validation labels
YValidation = imdsVal.Labels;

% Calculate final ensemble accuracy
ensembleAccuracy = sum(finalPredictions == YValidation) / numel(YValidation);
fprintf('Ensemble Validation Accuracy: %.2f%%\n', ensembleAccuracy * 100);

% Display Confusion Matrix
figure;
confusionchart(YValidation, finalPredictions);
title('Ensemble Model Confusion Matrix');
%% %% Step 1: Load and Modify GoogLeNet
net = googlenet;
inputSize = net.Layers(1).InputSize; % Get input size

% Convert to Layer Graph
lgraph = layerGraph(net);

% Display all layers (to confirm names)
disp({lgraph.Layers.Name}');

%% Step 2: Modify Final Layers for 12 Classes
% Create new layers with unique names to avoid conflicts
newFC = fullyConnectedLayer(12, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newClassOutput = classificationLayer('Name', 'new_output');

% Replace old layers with new ones
lgraph = replaceLayer(lgraph, "loss3-classifier", newFC);
lgraph = replaceLayer(lgraph, "prob", newSoftmax);
lgraph = replaceLayer(lgraph, "output", newClassOutput);

%% Step 3: Load & Augment Training Data
% Load the dataset
dataDir = fullfile('speechImageData'); 

imdsTrain = imageDatastore(fullfile(dataDir, 'TrainData'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsVal = imageDatastore(fullfile(dataDir, 'ValData'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Define augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXTranslation', [-3 3], ...
    'RandYTranslation', [-3 3], ...
    'RandXScale', [0.9 1.1], ...
    'RandYScale', [0.9 1.1]);

% Apply augmentation to training images
imageSize = [inputSize(1) inputSize(2) 3]; % GoogLeNet requires RGB images
% Function to Convert Grayscale to RGB
convertToRGB = @(img) cat(3, img, img, img);

% Apply transformation during image loading
augimdsTrain = augmentedImageDatastore([224 224 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore([224 224 3], imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');


%% Step 4: Define Training Options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ... % Faster training
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0001, ... % Small learning rate for fine-tuning
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

%% Step 5: Train the Transfer Learning Model
netTransfer = trainNetwork(augimdsTrain, lgraph, options);

%% % Load ResNet50
net = resnet50;
inputSize = net.Layers(1).InputSize; % Get input size

% Convert to Layer Graph
lgraph = layerGraph(net);

% Display all layers (to confirm names)
disp({lgraph.Layers.Name}');

% Find the correct names of the layers to replace
analyzeNetwork(net);
%% % Create new layers with unique names
newFC = fullyConnectedLayer(12, 'Name', 'new_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
newSoftmax = softmaxLayer('Name', 'new_softmax');
newClassOutput = classificationLayer('Name', 'new_output');

% Replace old layers
lgraph = replaceLayer(lgraph, "fc1000", newFC);  % Fully Connected
lgraph = replaceLayer(lgraph, "fc1000_softmax", newSoftmax);  % Softmax
lgraph = replaceLayer(lgraph, "ClassificationLayer_fc1000", newClassOutput);  % Classification
%% % Advanced Data Augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ... % Flip images horizontally
    'RandRotation', [-20 20], ... % Increase rotation range
    'RandScale', [0.7 1.3], ... % Increase scale variation
    'RandXTranslation', [-10 10], ...
    'RandYTranslation', [-10 10]);


% Convert Grayscale to RGB
convertToRGB = @(img) cat(3, img, img, img);

% Apply augmentation to training images
augimdsTrain = augmentedImageDatastore([224 224 3], imdsTrain, ...
    'DataAugmentation', imageAugmenter, 'ColorPreprocessing', 'gray2rgb');

augimdsVal = augmentedImageDatastore([224 224 3], imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

disp('Data Augmentation Applied');

%%
 options = trainingOptions('adam', ...
    'MaxEpochs', 50, ... % Increase from 30 to 50
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 0.0005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augimdsVal, ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ExecutionEnvironment', 'cpu');

  %% % Train ResNet50
netTransfer = trainNetwork(augimdsTrain, lgraph, options);
%% % Predict on validation data
YPred = classify(netTransfer, imdsVal);
YValidation = imdsVal.Labels;

% Calculate accuracy
transferAccuracy = sum(YPred == YValidation) / numel(YValidation);
fprintf('ResNet50 Validation Accuracy: %.2f%%\n', transferAccuracy * 100);

% Display confusion matrix
figure;
confusionchart(YValidation, YPred);
title('ResNet50 Confusion Matrix');

%% YPred = classify(net, imdsVal);
YValidation = imdsVal.Labels;
confusionchart(YValidation, YPred);
title('Baseline CNN Confusion Matrix');
%% % After training, get accuracy/loss data
trainingInfo = net.TrainingHistory;
figure;
plot([trainingInfo.TrainingAccuracy], 'LineWidth', 2); hold on;
plot([trainingInfo.ValidationAccuracy], 'LineWidth', 2);
legend('Training Accuracy', 'Validation Accuracy');
xlabel('Iteration');
ylabel('Accuracy');
title('Training Curve - Best Hyperparameter Configuration');


















