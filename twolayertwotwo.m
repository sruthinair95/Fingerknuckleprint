clc
close all

traindata= fullfile(matlabroot,'traindataset');
trainKnuckleData = imageDatastore(traindata,'IncludeSubfolders',true,'LabelSource','foldernames');
trainKnuckleData.countEachLabel;

trainingNumFiles = 3;
rng(1) % For reproducibility
[trainDigitData,testDigitData] = splitEachLabel(trainKnuckleData,...
				trainingNumFiles,'randomize');
            
            
 layers = [imageInputLayer([180 180 3],'DataAugmentation','randcrop')
          convolution2dLayer(7,22,'Stride',1,'padding',3)
          reluLayer
          convolution2dLayer(7,22,'Stride',1,'padding',3)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          convolution2dLayer(7,22,'Stride',1,'padding',3)
          reluLayer
          convolution2dLayer(7,22,'Stride',1,'padding',3)
          reluLayer
          maxPooling2dLayer(2,'Stride',2)
          fullyConnectedLayer(50)
          softmaxLayer
          classificationLayer()];
      
      
   functions = { ...
    @plotTrainingAccuracy, ...
    @(info) stopTrainingAtThreshold(info,95)};
    
    options = trainingOptions('sgdm',...
             'InitialLearnRate',0.0001,...
             'MaxEpochs',800,...
             'OutputFcn',functions);
       
  %'MaxEpochs',22,...
        
        
  convnet = trainNetwork(trainKnuckleData,layers,options);
save twolayertwotwo.mat convnet;





YTest = classify(convnet,testDigitData);
TTest = testDigitData.Labels;

accuracy = sum(YTest == TTest)/numel(TTest);




function plotTrainingAccuracy(info)

persistent plotObj

if info.State == "start"
    plotObj = animatedline;
    xlabel("Iteration")
    ylabel("Training Accuracy")
elseif info.State == "iteration"
    addpoints(plotObj,info.Iteration,info.TrainingAccuracy)
    drawnow limitrate nocallbacks
end

end

function stop = stopTrainingAtThreshold(info,thr)

stop = false;
if info.State ~= "iteration"
    return
end

persistent iterationAccuracy

% Append accuracy for this iteration
iterationAccuracy = [iterationAccuracy info.TrainingAccuracy];

% Evaluate mean of iteration accuracy and remove oldest entry
if numel(iterationAccuracy) == 50
    stop = mean(iterationAccuracy) > thr;

    iterationAccuracy(1) = [];
end

end





