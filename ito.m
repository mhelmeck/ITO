clc;
close all;
clear all;

neuron1 = struct('wagaNachylenia',51.0,'wagaDlugosci',460,'wagaOstrosci',70,'indexyCech',[]);
neuron2 = struct('wagaNachylenia',52.0,'wagaDlugosci',470,'wagaOstrosci',80,'indexyCech',[]);
neuron3 = struct('wagaNachylenia',53.0,'wagaDlugosci',480,'wagaOstrosci',90,'indexyCech',[]);
neuron4 = struct('wagaNachylenia',54.0,'wagaDlugosci',500,'wagaOstrosci',100,'indexyCech',[]);
neuron5 = struct('wagaNachylenia',55.0,'wagaDlugosci',510,'wagaOstrosci',110,'indexyCech',[]);
neuron6 = struct('wagaNachylenia',56.0,'wagaDlugosci',520,'wagaOstrosci',120,'indexyCech',[]);
neuron7 = struct('wagaNachylenia',57.0,'wagaDlugosci',530,'wagaOstrosci',130,'indexyCech',[]);
neuron8 = struct('wagaNachylenia',58.0,'wagaDlugosci',540,'wagaOstrosci',140,'indexyCech',[]);
% neuron1 = struct('wagaNachylenia',89.0,'wagaDlugosci',48,'wagaOstrosci',0.2,'indexyCech',[]);
% neuron2 = struct('wagaNachylenia',19.0,'wagaDlugosci',39,'wagaOstrosci',0.1,'indexyCech',[]);
% neuron3 = struct('wagaNachylenia',13.0,'wagaDlugosci',146,'wagaOstrosci',0.1,'indexyCech',[]);
% neuron4 = struct('wagaNachylenia',19.0,'wagaDlugosci',552,'wagaOstrosci',0.08,'indexyCech',[]);
% neuron5 = struct('wagaNachylenia',19.0,'wagaDlugosci',301,'wagaOstrosci',0.05,'indexyCech',[]);
% neuron6 = struct('wagaNachylenia',33.0,'wagaDlugosci',874,'wagaOstrosci',0.07,'indexyCech',[]);
% neuron7 = struct('wagaNachylenia',6.0,'wagaDlugosci',1394,'wagaOstrosci',0.08,'indexyCech',[]);
% neuron8 = struct('wagaNachylenia',7.0,'wagaDlugosci',439,'wagaOstrosci',0.09,'indexyCech',[]);
neurons = [neuron1, neuron2, neuron3, neuron4, neuron5, neuron6, neuron7, neuron8];

% myFolder = '/Users/maciejhelmecki/Desktop/LearningSet';
% if ~isdir(myFolder)
%     errorMessage = sprintf('Error: The following folder does not exist:\n%s', myFolder);
%     uiwait(warndlg(errorMessage));
%     return;
% end
% filePattern = fullfile(myFolder, '*.jpg');
% jpegFiles = dir(filePattern);
% for k = 1:length(jpegFiles)
%     baseFileName = jpegFiles(k).name;
%     fullFileName = fullfile(myFolder, baseFileName);
%     fprintf(1, 'Now reading %s\n', fullFileName);
%  
%     image = get_image(fullFileName);
%     feature = get_features(image);
%     result = ucz_siec(feature, neurons, false);
%     neurons = result;
% end

image = get_image('/Users/maciejhelmecki/Desktop/ITO_example_33.jpg');
feature = get_features(image);
result = ucz_siec(feature, neurons, true);
neurons = result;

% draw(myImageRGB_4)

% MARK: - Functions
function image = get_image(name) 
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
    disp(summedPointsGrayRate)
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

function draw(myImageRGB)
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