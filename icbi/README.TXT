README
============================================================

Name:
	ICBI (Iteractive Curvature Based Interpolation)

Version:
	1.1
	
Revision:
	000

Release date:
	2008-01-30

Author:
	Adrea Giachetti, Nicola Asuni 
	
Copyright (c) 2006-2008:
	Andrea Giachetti, andrea@andreagiachetti.it, via 24 Maggio 17 38100 Trento, Italy
	Nicola Asuni, nicola.asuni@tecnick.com

Software License:
	GNU GENERAL PUBLIC LICENSE v.2
	http://www.gnu.org/copyleft/gpl.html
	
Description:

icbi.m

% DESCRIPTION
    % --------------------
    % ICBI (Iteractive Curvature Based Interpolation)
    % This function returns an enlarged image by a factor 2^N implements
    % the enlargement methods FCBI and ICBI described in the paper
    % ""Fast artifacts-free image interpolation" presented at BMVC 2008.
    
    
    % KEYWORDS
    % --------------------
    % ICBI, image, zooming, magnification, upsizing, resampling,
    % resolution enhancement, interpolation, 
    % matlab, octave.
        
    
    % USAGE
    % --------------------
    % [EI] = icbi(IM)
    % [EI] = icbi(IM, ZK)
    % [EI] = icbi(IM, ZK, SZ)
    % [EI] = icbi(IM, ZK, SZ, PF)
    % [EI] = icbi(IM, ZK, SZ, PF, VR)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC, SC)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC, SC, TS)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC, SC, TS, AL)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC, SC, TS, AL, BT)
    % [EI] = icbi(IM, ZK, SZ, PF, VR, ST, TM, TC, SC, TS, AL, BT, GM)
    
    
    % INPUT
    % --------------------
    % IM : Source image.
    % ZK : Power of the zoom factor (default = 1)
    %      the image enlargement on vertical and horizontal direction is
    %      2^ZK; the final image size will be (SIZE * 2^ZK) - (2^ZK - 1).
    % SZ : Number of image bits per layer (default = 8).
    % PF : Potential to be minimized (default = 1).
    % VR : Verbose mode, if true prints some information during calculation
    %      (default = false).
    % ST : Maximum number of iterations (default = 20).
    % TM : Maximum edge step (default = 100).
    % TC : Edge continuity threshold (deafult = 50).
    % SC : Stopping criterion: 1 = change under threshold, 0 = ST iterations (default = 1).
    % TS : Threshold on image change for stopping iterations (default = 100).
    % AL : Weight for Curvature Continuity energy (default = 1.0).
    % BT : Weight for Curvature enhancement energy (default = -1.0).
    % GM : Weight for Isophote smoothing energy (default = 5.0).
    
    
    % OUTPUT
    % --------------------
    % EI : Enlarged image.
    
    
    % Examples
    % --------------------
    % Please check the icbiexample.m file on how to use this function.
    
    
    % Notes
    % --------------------
    % This implementation is not intended to be used in a production
    % environment. The main purpose of this script is to clearly show how
    % this technique works. Better performaces could be obtained using a
    % compiled version or rewriting this technique using a low-level
    % programming language.
    
    
    % ---------------------------------------------------------------------
    
============================================================
