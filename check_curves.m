 % Load video
vidObj = VideoReader('Untitled 720p.mp4');
numFrames = vidObj.NumFrames;

% Parameters for finding wingtip
wingLength = 50;
threshold = 0.8;

% Preallocate arrays for wingtip coordinates
xCoords = zeros(numFrames,1);
yCoords = zeros(numFrames,1);

% Loop over frames and track wingtip
for i = 1:numFrames
    % Read frame
    frame = read(vidObj,i);
    
    % Find wingtip
    [x,y] = findWingtip(frame, wingLength, threshold);
    
    % Save coordinates
    xCoords(i) = x;
    yCoords(i) = y;
end

% Find start and end frames for one flapping cycle
startFrame = 50;
endFrame = 63;

% Get coordinates of wingtip trajectory for one cycle
xCoordsCycle = xCoords(startFrame:endFrame);
yCoordsCycle = yCoords(startFrame:endFrame);

% Fit ellipse to trajectory points using DLS method
[x0,y0,a,b,phi] = fitEllipseDLS(xCoordsCycle,yCoordsCycle);

% Extract major and minor axis lengths
majorAxis = max(a,b);
minorAxis = min(a,b);

% Display major and minor axis lengths
disp(['Major axis length: ', num2str(majorAxis)]);
disp(['Minor axis length: ', num2str(minorAxis)]);

function [x,y] = findWingtip(frame, wingLength, threshold)
    % Convert to grayscale and equalize histogram
    grayFrame = rgb2gray(frame);
    grayFrame = histeq(grayFrame);
    
    % Find edges and dilate them
    edges = edge(grayFrame, 'canny');
    se = strel('disk', wingLength);
    dilatedEdges = imdilate(edges, se);
    
    % Find centroid of top connected component
    cc = bwconncomp(dilatedEdges);
    stats = regionprops(cc, 'Centroid');
    centroids = cat(1,stats.Centroid);
    [~,idx] = max(centroids(:,2));
    x = centroids(idx,1);
    y = centroids(idx,2);
    
    % Check if confidence is high enough
    confidence = sum(sum(dilatedEdges))/sum(sum(edges));
    if confidence < threshold
        x = NaN;
        y = NaN;
    end
end


function [x0,y0,a,b,phi] = fitEllipseDLS(x,y)
%FITELLIPSEDLS Fit an ellipse to a set of 2D points using the Direct Least
%Squares (DLS) method.
%
%   [X0,Y0,A,B,PHI] = FITELLIPSEDLS(X,Y) fits an ellipse to the set of 2D
%   points specified by X and Y using the Direct Least Squares (DLS)
%   method. X and Y are column vectors of the same length, and represent
%   the x and y coordinates of the points, respectively. The function
%   returns the center coordinates (X0,Y0), the major and minor axis
%   lengths A and B, and the orientation angle PHI (in radians) of the
%   fitted ellipse.
%
%   References:
%
%   [1] Fitzgibbon, Pilu, and Fisher. "Direct least squares fitting of
%   ellipses." IEEE Transactions on Pattern Analysis and Machine
%   Intelligence, Vol. 21, No. 5, May 1999.
%
%   [2] J. Zheng, S. Cui, X. Chen, and W. Li, "Least squares fitting of
%   ellipse parameters from contour points," Journal of Computational
%   Information Systems, vol. 6, no. 4, pp. 1274-1281, 2010.
%
%   Author:  Damian Trilling
%   Website: https://www.damiantrilling.net
%   Version: 1.0
%   Date:    2019-05-08
%   License: MIT License (see LICENSE file in repository root)



% Convert input to column vectors
x = x(:);
y = y(:);

% Formulate the problem as a linear system of the form Ax = b
A = [x.^2, x.*y, y.^2, x, y, ones(size(x))];
b = ones(size(x));

% Solve the linear system using the SVD method
[~,~,V] = svd(A,0);
x = V(:,1);
x = x / x(6);

% Extract ellipse parameters
a = sqrt(2/x(1));
b = sqrt(2/x(3));
c = sqrt(1/x(2)^2 - 1/x(1)/x(3));
x0 = (x(3)*x(4) - x(2)*x(5)) / (x(2)^2 - x(1)*x(3));
y0 = (x(1)*x(5) - x(2)*x(4)) / (x(2)^2 - x(1)*x(3));
phi = 0.5 * atan2(2*x(2), x(1)-x(3));

% Ensure that major axis length is always greater than or equal to minor
% axis length
if b > a
    temp = a;
    a = b;
    b = temp;
    phi = phi + pi/2;
end
% Compute the eccentricity of the ellipse
midpoint = [mean(x), mean(y)];
endpoint1 = [x(1), y(1)];
endpoint2 = [x(end), y(end)];

% Compute the value of k
k = ((midpoint - endpoint1) - (midpoint - endpoint2)) / 2;
disp(k);
% Classify the shape of the trajectory based on the value of k
if abs(k) < 0.01
    fprintf('The trajectory is an ellipse.\n');
elseif k >= 0.01
    fprintf('The trajectory is an offset ellipse.\n');
else
    fprintf('The trajectory is neither an ellipse nor an offset ellipse.\n');
end

end
