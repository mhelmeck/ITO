clc;
close all;
clear all;

neuron_1 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_2 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_3 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_4 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_5 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_6 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_7 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);
neuron_8 = struct('distanceWeight',0,'slopeWeight',0,'clarityWeight',0,'arrayOfLineIndex',[]);

neurons = [neuron_1, neuron_2, neuron_3, neuron_4, neuron_5, neuron_6, neuron_7, neuron_8];

% folder = '/Users/maciejhelmecki/Desktop/LearningSet';
% if ~isfolder(folder)
%     errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
%     uiwait(warndlg(errorMessage));
%     return;
% end
% filePattern = fullfile(folder, '*.jpg');
% jpegFiles = dir(filePattern);
% for index = 1:length(jpegFiles)
%     baseFileName = jpegFiles(index).name;
%     fullFileName = fullfile(folder, baseFileName);
%     fprintf(1, 'Now reading %s\n', fullFileName);
%  
%     rgbImage = get_rgb_image(fullFileName);
%     imageFeatures = get_features(rgbImage);
%     updatedNeurons = train_network(imageFeatures, neurons, false);
%     neurons = updatedNeurons;
% end

rgbImage = get_rgb_image('/Users/maciejhelmecki/Desktop/ITO_example_24.jpg');
imageFeatures = get_features(rgbImage);
updatedNeurons = train_network(imageFeatures, neurons, true);
neurons = updatedNeurons;

draw(rgbImage, neurons(2).arrayOfLineIndex)

% MARK: - Functions
function image = get_rgb_image(name) 
    myImageRGB = imread(name);
    image = imresize(myImageRGB,[1200, 1920]);
end

function features = get_features(myImageRGB)
    myImageGray = rgb2gray(myImageRGB);
    BW = edge(myImageGray,'prewitt', 0.11);

    [H,T,R] = hough(BW);

    P = houghpeaks(H,50,'threshold',ceil(0.1*max(H(:))));
    lines = houghlines(BW,T,R,P,'FillGap',30,'MinLength',10);

    for lineIndex = 1:length(lines)
        temp(lineIndex) = get_feature(lines(lineIndex), myImageGray); 
    end
    
    features = temp;
end

function updatedNeurons = train_network(features, neurons, zapamietajIndexyCech)
    mi = 0.3;
    for k = 1:length(features)
        feature = features(k);

        minEuklides = 1000000;
        wonNeuronIndex = 0;
        for i = 1:length(neurons)
            neuron = neurons(i);

            euklidesLini = get_euklides_value(feature.slope, neuron.slopeWeight, feature.distance, neuron.distanceWeight, feature.clarity, neuron.clarityWeight);
            if euklidesLini < minEuklides
               minEuklides = euklidesLini;
               wonNeuronIndex = i;
            end
        end
        wonNeuron = neurons(wonNeuronIndex);
        temp = struct('distanceWeight',feature.distance - wonNeuron.distanceWeight,'slopeWeight',feature.slope - wonNeuron.slopeWeight,'clarityWeight',feature.clarity - wonNeuron.clarityWeight);
        miTemp = struct('distanceWeight',mi * temp.distanceWeight,'slopeWeight',mi * temp.slopeWeight,'clarityWeight',mi * temp.clarityWeight);
       
        if zapamietajIndexyCech
            test = wonNeuron.arrayOfLineIndex;
            testLength = length(test);
            test(testLength + 1) = k;
        else 
            test = [];
        end
        
        updatedNeuron = struct('distanceWeight',wonNeuron.distanceWeight + miTemp.distanceWeight,'slopeWeight',wonNeuron.slopeWeight + miTemp.slopeWeight,'clarityWeight',wonNeuron.clarityWeight + miTemp.clarityWeight,'arrayOfLineIndex',test);
        neurons(wonNeuronIndex) = updatedNeuron;
                
        updatedNeurons = neurons;
    end
end

function euklides = get_euklides_value(slope, slopeWeight, distance, distanceWeight, clarity, clarityWeight)
    sumOfSquares = (slope - slopeWeight)^2 + (distance - distanceWeight)^2 + (clarity - clarityWeight)^2 ;
    euklides = sqrt(sumOfSquares);
end

function feature = get_feature(line, image)
    x_start = line.point1(1);
    y_start = line.point1(2);
    x_end = line.point2(1);
    y_end = line.point2(2);

    startPoint = struct('x',x_start,'y',y_start);
    endPoint = struct('x',x_end,'y',y_end);
    slope = abs((atan2(y_end - y_start, x_end - x_start)) * 180 / pi);
    distance = sqrt((x_end - x_start)^2 + (y_end - y_start)^2);
    clarity = get_average_gray_rate(x_start, y_start, x_end, y_end, image);
    
    feature = struct('startPoint',startPoint,'endPoint',endPoint,'distance',distance,'slope',slope,'clarity',clarity);
end

function averageGrayRate = get_average_gray_rate(x_start, y_start, x_end, y_end, image)
    points_between = get_points(x_start, y_start, x_end, y_end);
    pointsGrayRate = [];
    imageSize = size(image);
    width = imageSize(1);
    height = imageSize(2);
    for index = 1:length(points_between)
        point = points_between(index);
        if point.x < width && point.y < height
            pointGrayRate = image(point.x,point.y);
            pointsGrayRate(index) = pointGrayRate;
        end
    end
    summedPointsGrayRate = sum(pointsGrayRate);
    averageGrayRate = summedPointsGrayRate / length(points_between);
end

function pointBetween = get_points(x_start, y_start, x_end, y_end)
    steep = abs(y_end - y_start) > abs(x_end - x_start);
    x_0 = x_start;
    y_0 = y_start;
    x_1 = x_end;
    y_1 = y_end;
    if steep
        temp = x_0;
        x_0 = y_0;
        y_0 = temp;
        temp = x_1;
        x_1 = y_1;
        y_1 = temp;
    end
    if x_0 > x_1
        temp = x_0;
        x_0 = x_1;
        x_1 = temp;
        temp = y_0;
        y_0 = y_1;
        y_1 = temp;
    end
    difference_y = abs(y_1 - y_0);
    difference_x = x_1 - x_0;
    error = difference_x / 2;
    
    if y_0 < y_1
        step_y = 1;
    else
        step_y = -1;
    end
    y = y_0;
    
    
    for z = 1:(x_1 + 1 - x_0)
        x = z + x_0 - 1;
        
        if steep
            point1 = struct('x',y,'y',x);
            pointBetween(z) = point1;
        else
            point2 = struct('x',x,'y',y);
            pointBetween(z) = point2;
        end
        
        error = error - difference_y;
        if error < 0
            y = y + step_y;
            error = error + difference_x;
        end
    end
end

function draw(myImageRGB, arrayOfLineIndex)
    myImageGray = rgb2gray(myImageRGB);
    BW = edge(myImageGray,'prewitt', 0.11);

    [H,T,R] = hough(BW);

    P = houghpeaks(H,50,'threshold',ceil(0.1*max(H(:))));
    lines = houghlines(BW,T,R,P,'FillGap',30,'MinLength',10);
    disp(length(lines))
    
    figure, imshow(BW), hold on
    max_len = 0;
    for k = 1:length(lines)
        xy = [lines(k).point1; lines(k).point2];
        
        
        % Plot beginnings and ends of lines
        if ismember(k, arrayOfLineIndex)
%             plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','red');
%             plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');
        else
            plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','blue');
%             plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','blue');
%             plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','blue');
        end

       % Determine the endpoints of the longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ( len > max_len)
          max_len = len;
       end
    end
end