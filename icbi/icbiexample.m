function [] = icbiexample()
    % =====================================================================
    % File name   : icbiexample.m
    % File Type   : m-file (script file for Matlab or Octave)
    % Requirements: Image Processing Toolbox for Matlab or 
    %               ImageMagick for Octave
    % Begin       : 2006-07-07
    % Last Update : 2008-01-30
    % Author      : Andrea Giachetti, Nicola Asuni
    % Description : Example script for inedi.m usage.
    % Copyright   : Andrea Giachetti, andrea@andreagiachetti.it,
    %               Nicola Asuni, nicola.asuni@tecnick.com           
    % License     : GNU GENERAL PUBLIC LICENSE v.2
    %               http://www.gnu.org/copyleft/gpl.html
    % Version     : 1.1
    % =====================================================================

    % DESCRIPTION
    % --------------------
    % This is an example script that shows you how to use the icbi.m function.
    %
    % The icbi.m function returns an enlarged image by a factor of 2^ZK for
    % both horizontal and vertical directions. 

    % USAGE
    % --------------------
    % icbiexample

    % Example
    % --------------------
    % >> icbiexample

    % ---------------------------------------------------------------------

    % Help
    disp('ICBI function example.');
    disp('This is an example script that shows you how to use the icbi.m function.');
    disp('Please check the documentation inside this file for further information.');

    % --- set parameters ---

    % Power of the zoom factor.
    % The image enlargement on vertical and horizontal direction is 2^ZK.
    ZK = 1;
    
    % --- end set parameters ---

    % load test image
    IM = imread('testimage.png');

    % display original image
    figure, imshow(IM);

    % get the enlarged image
    [EI] = icbi(IM, ZK, 8, 1, true);

    % save image to disk
    % imwrite(EI,'testimage_icbi.tif','tif');

    % display enlarged image
    figure, imshow(EI);

% === EOF =================================================================