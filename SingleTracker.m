%%Part2

SingleTracker('Football2.avi');

function SingleTracker(file_name)
%prompt = 'What is the file name? (including file extension)';
%prompt = 'What is the file name? (including file extension)';
%file_name = input(prompt);
showDetections();
showTrajectory();
frame            = [];
detectedLocation = [];
trackedLocation  = [];
label            = '';
utilities        = [];
function trackSingleObject(param)
  utilities = createUtilities(param);
  isTrackInitialized = false;
  while ~isDone(utilities.videoReader)
    frame = readFrame();
    [detectedLocation, isObjectDetected] = detectObject(frame);

    if ~isTrackInitialized
      if isObjectDetected
        initialLocation = computeInitialLocation(param, detectedLocation);
        kalmanFilter = configureKalmanFilter(param.motionModel, ...
          initialLocation, param.initialEstimateError, ...
          param.motionNoise, param.measurementNoise);

        isTrackInitialized = true;
        trackedLocation = correct(kalmanFilter, detectedLocation);
        label = 'Initial';
      else
        trackedLocation = [];
        label = '';
      end

    else
      if isObjectDetected
        predict(kalmanFilter);
        trackedLocation = correct(kalmanFilter, detectedLocation);
        label = 'Corrected';
      else
        trackedLocation = predict(kalmanFilter);
        label = 'Predicted';
      end
    end

    annotateTrackedObject();
  end
  
  showTrajectory();
end


param = getDefaultParameters();

trackSingleObject(param); 

function param = getDefaultParameters
  param.motionModel           = 'ConstantAcceleration';
  param.initialLocation       = 'Same as first detection';
  param.initialEstimateError  = 1E5 * ones(1, 3);
  param.motionNoise           = [25, 10, 1];
  param.measurementNoise      = 1250;
  param.segmentationThreshold = 0.005;
end

function frame = readFrame()
  frame = step(utilities.videoReader);
end

function showDetections()
  param = getDefaultParameters();
  utilities = createUtilities(param);
  trackedLocation = [];

  idx = 0;
  while ~isDone(utilities.videoReader)
    frame = readFrame();
    detectedLocation = detectObject(frame);
    annotateTrackedObject();
    idx = idx + 1;
    if idx == 200
      combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), frame);
    end
  end

  uiscopes.close('All'); 
end

function [detection, isObjectDetected] = detectObject(frame)
  grayImage = rgb2gray(frame);
  utilities.foregroundMask = step(utilities.foregroundDetector, grayImage);
  detection = step(utilities.blobAnalyzer, utilities.foregroundMask);
  if isempty(detection)
    isObjectDetected = false;
  else
    detection = detection(1, :);
    isObjectDetected = true;
  end
end

function annotateTrackedObject()
  accumulateResults();
  combinedImage = max(repmat(utilities.foregroundMask, [1,1,3]), frame);

  if ~isempty(trackedLocation)
    shape = 'circle';
    region = trackedLocation;
    region(:, 3) = 5;
    combinedImage = insertObjectAnnotation(combinedImage, shape, ...
      region, {label}, 'Color', 'red');
  end
  step(utilities.videoPlayer, combinedImage);
end

function showTrajectory

  uiscopes.close('All'); 

  figure; imshow(utilities.accumulatedImage/2+0.5); hold on;
  plot(utilities.accumulatedDetections(:,1), ...
    utilities.accumulatedDetections(:,2), 'k+');
  
  if ~isempty(utilities.accumulatedTrackings)
    plot(utilities.accumulatedTrackings(:,1), ...
      utilities.accumulatedTrackings(:,2), 'r-o');
    legend('Detection', 'Prediction');
  end
end

function accumulateResults()
  utilities.accumulatedImage      = max(utilities.accumulatedImage, frame);
  utilities.accumulatedDetections ...
    = [utilities.accumulatedDetections; detectedLocation];
  utilities.accumulatedTrackings  ...
    = [utilities.accumulatedTrackings; trackedLocation];
end

function loc = computeInitialLocation(param, detectedLocation)
  if strcmp(param.initialLocation, 'Same as first detection')
    loc = detectedLocation;
  else
    loc = param.initialLocation;
  end
end

function utilities = createUtilities(param)
  
  utilities.videoReader = vision.VideoFileReader(file_name);
  utilities.videoPlayer = vision.VideoPlayer('Position', [100,100,500,400]);
  utilities.foregroundDetector = vision.ForegroundDetector(...
    'NumTrainingFrames', 10, 'InitialVariance', param.segmentationThreshold);
  utilities.blobAnalyzer = vision.BlobAnalysis('AreaOutputPort', false, ...
    'MinimumBlobArea', 50, 'CentroidOutputPort', true);

  utilities.accumulatedImage      = 0;
  utilities.accumulatedDetections = zeros(0, 2);
  utilities.accumulatedTrackings  = zeros(0, 2);
end

end
