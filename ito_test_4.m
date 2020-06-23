clc;
close all;
clear all;

neuron1 = struct('wagaNachylenia',90.0,'wagaDlugosci',900,'wagaOstrosci',200,'indexyCech',[]);
neuron2 = struct('wagaNachylenia',0.0,'wagaDlugosci',900,'wagaOstrosci',200,'indexyCech',[]);
neuron3 = struct('wagaNachylenia',90.0,'wagaDlugosci',300,'wagaOstrosci',200,'indexyCech',[]);
neuron4 = struct('wagaNachylenia',0.0,'wagaDlugosci',300,'wagaOstrosci',200,'indexyCech',[]);
neuron5 = struct('wagaNachylenia',90.0,'wagaDlugosci',300,'wagaOstrosci',90,'indexyCech',[]);
neuron6 = struct('wagaNachylenia',0.0,'wagaDlugosci',300,'wagaOstrosci',90,'indexyCech',[]);
neuron7 = struct('wagaNachylenia',90.0,'wagaDlugosci',900,'wagaOstrosci',90,'indexyCech',[]);
neuron8 = struct('wagaNachylenia',0.0,'wagaDlugosci',900,'wagaOstrosci',90,'indexyCech',[]);
neurons = [neuron1, neuron2, neuron3, neuron4, neuron5, neuron6, neuron7, neuron8];

% myImage_1 = get_bw_image('/Users/maciejhelmecki/Desktop/ITO_example_1.jpg');
% myImage_2 = get_bw_image('/Users/maciejhelmecki/Desktop/ITO_example_2.jpg');
% myImage_3 = get_bw_image('/Users/maciejhelmecki/Desktop/ITO_example_3.jpg');
% myImage_4 = get_bw_image('/Users/maciejhelmecki/Desktop/ITO_example_4.jpg');

% images = [myImage_1, myImage_2, myImage_3, myImage_4];
% image = images(1);
% % for i = 1:length(images)
% %     image = images(i);
% % %     test = get_features(image);
% % end

myFolder = '/Users/maciejhelmecki/Desktop/LearningSet';
if ~isdir(myFolder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
    uiwait(warndlg(errorMessage));
    return;
end
filePattern = fullfile(myFolder, '*.jpg');
jpegFiles = dir(filePattern);
for k = 1:length(jpegFiles)
    baseFileName = jpegFiles(k).name;
    fullFileName = fullfile(myFolder, baseFileName);
    fprintf(1, 'Now reading %s\n', fullFileName);
 
    image = get_bw_image(fullFileName);
    feature = get_features(image);
    test = ucz_siec(feature, neurons, false);
    neurons = test;
end

% image2 = get_bw_image('/Users/maciejhelmecki/Desktop/ITO_example_4.jpg');
% feature2 = get_features(image2);
% test = ucz_siec(feature2, neurons, true);
% neurons = test;

% features_1 = get_features(myImage_1);
% features_2 = get_features(myImage_2);
% features_3 = get_features(myImage_3);
% features_4 = get_features(myImage_4); 
% 
% test_1 = ucz_siec(features_1, neurons, false);
% test_2 = ucz_siec(features_2, test_1, false);
% test_3 = ucz_siec(features_3, test_2, false);
% test_4 = ucz_siec(features_4, test_3, true);

% draw(myImageRGB_4)

% MARK: - Functions
function BW = get_bw_image(name) 
    myImageRGB = imread(name);
    resizedMyImageRGB = imresize(myImageRGB,[1200, 1920]);
    myImageGray = rgb2gray(resizedMyImageRGB);
    
    BW = edge(myImageGray,'prewitt', 0.11);
end

function features = get_features(BW)
%     myImageGray = rgb2gray(myImageRGB);
%     BW = edge(myImageGray,'prewitt', 0.11);

    [H,T,R] = hough(BW);

    P = houghpeaks(H,50,'threshold',ceil(0.1*max(H(:))));
    lines = houghlines(BW,T,R,P,'FillGap',30,'MinLength',10);

    for lineIndex = 1:length(lines)
        temp(lineIndex) = get_feature(lines(lineIndex), BW); 
    end
    
    features = temp;
end

function updated_neurons = ucz_siec(features, neurons, zapamietajIndexyCech)
    mi = 0.3;
    for k = 1:length(features)
        feature = features(k);

        minEuklides = 1000000;
        wonNeuronIndex = 0;
        for i = 1:length(neurons)
            neuron = neurons(i);

            euklidesLini = get_euklides_value(feature.theta, neuron.wagaNachylenia, feature.distance, neuron.wagaDlugosci, feature.lineGrayRate, neuron.wagaOstrosci);
            if euklidesLini < minEuklides
               minEuklides = euklidesLini;
               wonNeuronIndex = i;
            end
        end
        wonNeuron = neurons(wonNeuronIndex);
        temp = struct('wagaNachylenia',feature.theta - wonNeuron.wagaNachylenia,'wagaDlugosci',feature.distance - wonNeuron.wagaDlugosci,'wagaOstrosci',feature.lineGrayRate - wonNeuron.wagaOstrosci);
        miTemp = struct('wagaNachylenia',mi * temp.wagaNachylenia,'wagaDlugosci',mi * temp.wagaDlugosci,'wagaOstrosci',mi * temp.wagaOstrosci);
       
        if zapamietajIndexyCech
            test = wonNeuron.indexyCech;
            testLength = length(test);
            test(testLength + 1) = k;
        else 
            test = [];
        end
        
        updatedNeuron = struct('wagaNachylenia',wonNeuron.wagaNachylenia + miTemp.wagaNachylenia,'wagaDlugosci',wonNeuron.wagaDlugosci + miTemp.wagaDlugosci,'wagaOstrosci',wonNeuron.wagaOstrosci + miTemp.wagaOstrosci,'indexyCech',test);
        neurons(wonNeuronIndex) = updatedNeuron;
                
        updated_neurons = neurons;
    end
end

function euklides = get_euklides_value(nachylenie, wagaNachylenia, dlugosc, wagaDlugosci, ostrosc, wagaOstrosci)
    sumaKwadratow = (nachylenie - wagaNachylenia)^2 + (dlugosc - wagaDlugosci)^2 + (ostrosc - wagaOstrosci)^2 ;
    euklides = sqrt(sumaKwadratow);
end

function feature = get_feature(line, image)
    x_start = line.point1(1);
    y_start = line.point1(2);
    x_end = line.point2(1);
    y_end = line.point2(2);

    startPoint = struct('x',x_start,'y',y_start);
    endPoint = struct('x',x_end,'y',y_end);
    theta = abs((atan2(y_end - y_start, x_end - x_start)) * 180 / pi);
    distance = sqrt((x_end - x_start)^2 + (y_end - y_start)^2);
    lineGrayRate = get_average_gray_rate(x_start, y_start, x_end, y_end, image);
    
    feature = struct('startPoint',startPoint,'endPoint',endPoint,'theta',theta,'distance',distance,'lineGrayRate',lineGrayRate, 'class', 0);
end

function averageGrayRate = get_average_gray_rate(x_start, y_start, x_end, y_end, image)
    points_between = get_points(x_start, y_start, x_end, y_end);
    pointsGrayRate = [];
    imageSize = size(image);
    width = imageSize(1);
    height = imageSize(2);
    for i = 1:length(points_between)
        point = points_between(i);
        if point.x < width && point.y < height
            pointGrayRate = image(point.x,point.y);
            pointsGrayRate(i) = pointGrayRate;
        end
    end
    summedPointsGrayRate = sum(pointsGrayRate);
    averageGrayRate = summedPointsGrayRate / length(points_between);
end

function points_between = get_points(x_start, y_start, x_end, y_end)
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
            points_between(z) = point1;
        else
            point2 = struct('x',x,'y',y);
            points_between(z) = point2;
        end
        
        error = error - difference_y;
        if error < 0
            y = y + step_y;
            error = error + difference_x;
        end
    end
end

function draw(BW)
%     myImageGray = rgb2gray(myImageRGB);
%     BW = edge(myImageGray,'prewitt', 0.11);

    [H,T,R] = hough(BW);

    P = houghpeaks(H,50,'threshold',ceil(0.1*max(H(:))));
    lines = houghlines(BW,T,R,P,'FillGap',30,'MinLength',10);
    disp(length(lines))
    
    figure, imshow(BW), hold on
    max_len = 0;
    for k = 1:length(lines)
       xy = [lines(k).point1; lines(k).point2];
       plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red');

       % Plot beginnings and ends of lines
       plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','red');
       plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');

       % Determine the endpoints of the longest line segment
       len = norm(lines(k).point1 - lines(k).point2);
       if ( len > max_len)
          max_len = len;
          xy_long = xy;
       end
    end
end

% MARK: - TEST
% subplot(1,2,1);
% imshow(BW);
% title('BW');
% 
% imageSize = size(myImageGray)
% width = imageSize(1)
% height = imageSize(2)
% for i = 1:width
%     for j = 1:height
%         A(i,j) = false;
%     end
% end
% for k = 1:length(lines)
%     line = lines(k);
%     x_start = line.point1(1);
%     y_start = line.point1(2);
%     x_end = line.point2(1);
%     y_end = line.point2(2);
%     
%     points = get_points(x_start, y_start, x_end, y_end);
%     for i = 1:length(points)
%         point = points(i);
%         A(point.y,point.x) = true;
%     end
% end 
% subplot(1,2,2);
% imshow(A);
% title('A');