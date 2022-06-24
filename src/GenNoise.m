
% Faux Comets for Training Sets -Sid Shaw 2022
% V 1.0 22/16/05
%
% This is starter code for producing a 'comet' image representing the
% accumulation of fluorescent proteins on the end of a growing polymer
% where the effective binding site distribution is governed by an
% exponential to be highest at the tip.  The polymer is typically 25nm in
% diameter and has 13 binding sites distributed around the circumference in
% a slightly offset beta lattice.
%
% The simulated image is created in parts. A) We first create a model of the
% microtubule in 3D space with rotation and subpositioning in space at a
% specific resolution.  B) Next, the EB1-GFP molecules are added where they
% respect a probability distribution related to occupancy and state of the
% binding site.  C) We then distribute the photons that would be emitted in
% random directions and in amounts related to exposure time and emission
% rate.  D) We finally simpulate the camera by downsampling the photon
% spread into a cartesian pixel grid and simulating the camera-dependent
% noise.
%
% Variables governing the simulation come from user inputs for this example
% but they can be directly input as variables.
%
% - the actual size of the imaged area in nanometers
% ImSize  = str2double(get(data.ImSize,'string'));    % image size nm (2000)
% PolAng  = str2double(get(data.PolAng,'string'));    % polymer angle (30)
% PolCurv = str2double(get(data.PolCurv,'string'));   % polymer curvature(5)
% PolDiam = str2double(get(data.PolDiam,'string'));   % Polymer diameter(25)
% ProSize = 4;                                        % Eb1-GFP size in nm (4)
% LatProb = str2double(get(data.LatProb,'string'));   % Lattice basel perc(.1)
% TipExp  = str2double(get(data.TipExp,'string'));    % Exponent dist (200)
% TipProb = str2double(get(data.TipProb,'string'));   % Binding probability(.8)
% WaveL   = str2double(get(data.WaveL,'string'));     % Imaging wavelength(515)
% PhotSec = str2double(get(data.PhotSec,'string'));   % Mean photons / sec (500)
% PixSize = str2double(get(data.PixSize,'string'));   % pixel size at mag (66)
% SubPix  = str2double(get(data.SubPix,'string'));    % Degree of subpixel samp (5)
% eNoise  = str2double(get(data.eNoise,'string'));    % Electronic noise (Gaus) (50)
% eDepth  = str2double(get(data.eDepth,'string'));    % Depth of electron count (15000)
% ADCval  = str2double(get(data.ADCval,'string'));    % ADC value (45)
% ExpTime = str2double(get(data.ExpT,'string'));      % Camera exposure time (100 msec)
% IM      = zeros(ImSize,ImSize);                     % The final image
%


% Effective fake image size = ImSize/PixSize = 2000/64 ~ 31
% PixSize changed to 64
%
%
for snum = 1:200
    ImSize  = 3630;  % real space of specimen in nanometers
    PolAng  = randi([0 90],1,1);   
    PolCurv = 5;
    PolDiam = randi([25 250],1,1);
    ProSize = 4;                                        % Eb1-GFP size in nm (4)
    LatProb = .01;   % there is a chance of binding anywhere on the lattice
    TipExp  = randi([50 2500],1,1);   % Exponent for creating probability of this distance in nm
    TipProb = randi([500 900],1,1)/1000;    % Chance of being occupied if it is possible
    WaveL   = 515;
    PhotSec = randi([10 200],1,1);   % photons per second (mean of Poisson dist)
    PixSize = 66;    % the size the pixel would be at specimen in nanometers
    SubPix  = 3;     % number of x and y blocks to cut up the pixel position on tip
    eNoise  = randi([50 700],1,1);   % std of gaussian distributed noise from electronics per sec
    eDepth  = 15000; % maximum number of electrons in camera well
    ADCval  = 8;     % conversion factor for electrons to gray levels
    ExpTime = randi([50 150],1,1);   % camera exposure in milliseconds
    IM      = zeros(ImSize,ImSize);                     % The final image
    %
    %%%%%% A. Object Model for Eb1-GFP positions on lattice
    radi = (PolDiam+ProSize)/2; % center of MT to the radial positon of the Eb1
    incr = 8.5/12;              % 8.5 nm inc per 13 protofilaments
    subl = 8;                   % subunit length or distance along filament nm
    subw = 4;                   % subunit distance across the lattice nm
    maxc = floor(ImSize/2);     % maximum GTP cap length in nm
    maxs = round(maxc/subl);    % max subunits of cap length
    % Create a ring of Eb1 on Tub lattice
    capp = radi.*[sin(0:(2*pi)/13:(2*pi));cos(0:(2*pi)/13:(2*pi))]';
    capp = capp(1:13,:);        % single ring position minus seam
    rngl = 0:incr:8.5;          % increment along lattice
    % Create the array for 3D positions
    capm      = zeros(13*maxs,5);           % array of x,y,z,value for tubs
    capm(:,2) = repmat(capp(:,1),maxs,1);   % Y positions
    capm(:,3) = repmat(capp(:,2),maxs,1);   % Z positions
    capg      = repmat(0:subl:maxc,13,1);   % X positions
    for i = 1:13    % for each yz ring of tubulin on x
        capg(i,:) = capg(i,:)+rngl(i);
    end
    capg      = capg(:,1:maxs);
    capm(:,1) = reshape(capg,1,numel(capg))';
    % Rotation of the model in 2D
    th        = PolAng;
    R         = [cosd(th) -sind(th); sind(th) cosd(th)];
    capm(:,1:2) = capm(:,1:2)*R;  % final polymer model
    % Check tube in 2d
    % figure; plot(capm(:,1),capm(:,2),'.'); axis equal;
    %%%%%% B. Distribution Model for Eb1-GFP binding on lattice 
    % Exponental with distance from tip as probability of the site being
    % active or inactive for binding.  Then should have a probability of
    % binding if the site is high vs low.
    BP        = LatProb;       % Base probability of lattice occupation if not GTP
    Y         = unifpdf(capm(:,1), -TipExp, TipExp);% exp decay or go with prob of hydrolysis
    Y         = Y./max(Y);             % Probability that the site is open
    Y         = Y.*TipProb;            % Prob that site is occupied if open
    Y(Y<BP)   = BP;                    % Final probability
    capm(:,4) = Y;                     % Set the probability of binding per site
    capm(:,5) = rand(length(capm),1) <= Y;
    capm(:,1) = round(ImSize/2)+1-capm(:,1);
    capm(:,2) = round(ImSize/2)+capm(:,2);
    capm      = capm(capm(:,5)>0,:);
    capm      = capm(capm(:,1)>0,:);    % just keep the part that is emitting
    %%%%%% C. Optical image distribution of photons
    WL        = WaveL;                              % wavelength of emission
    PH        = PhotSec*(ExpTime/1000);             % MEAN PHOTON EMISSION
    PT        = random('poiss',PH,length(capm),1);  % # of photons
    PTc1      = [0; cumsum(PT)]+1;
    PTc2      = cumsum(PT);
    PTx       = zeros(sum(PT),2);
    R         = 0:1/((1/.61)*(WL/2)):1;             % Range for bessel
    f         = (besselj(1,(2*pi)*R(:))./R(:)).^2;  % bessel is lens function
    f(1)      = max(f);
    c         = cumsum(f./sum(f));                  % cdf of bessel
    % 2D photon distribution - how the distribute within lens function
    for i = 1:length(capm)
       rs = sum(c<sqrt(rand(1,PT(i)))); % as a distance from center point
       an = rand(1,PT(i)).*360;         % as an angle from point
       dx = round([capm(i,1)+(cosd(an).*rs); capm(i,2)+(sind(an).*rs)]);
       PTx(PTc1(i):PTc2(i),:) = dx';
    end
    % Trim out any photons  outside of the frame (less than 0)
    PTxd = PTx(PTx(:,1)>0 & PTx(:,2)>0,:);
    % Use matlab call to accumulate indexed positions to 2D counts (sums)
    IMX  = accumarray(PTxd,1);
    % Place these accumulated photons into the image matrix
    % figure
    SZ   = size(IMX);
    IM(1:SZ(1),1:SZ(2)) = IMX;
    % axes(data.TipAxis);
    ma = max(IM(:));
    % imagesc(IM',[0 ma]);
    % zoom on;
    imtube = 0;
    % This remakes the polymer for visualization only. Not needed to make the
    % image of the comet
    if imtube
        hold on;
        radi = 11;       % center of MT to the radial positon of the Eb1
        capp = radi.*[sin(0:(2*pi)/13:(2*pi));cos(0:(2*pi)/13:(2*pi))]';
        capp = capp(1:13,:);        % single ring position minus seam
        capb(:,2) = repmat(capp(:,1),maxs,1);   % Y positions
        capb(:,3) = repmat(capp(:,2),maxs,1);   % Z positions
        capa(:,2) = repmat(capp(:,1),maxs,1);   % Y positions
        capa(:,3) = repmat(capp(:,2),maxs,1);   % Z positions
        capg      = repmat(0:subl:maxc,13,1);   % X positions
        for i = 1:13    % for each yz ring of tubulin on x
            capg(i,:) = capg(i,:)+rngl(i);
        end
        capg      = capg(:,1:maxs);
        capb(:,1) = 4+reshape(capg,1,numel(capg))';
        capa(:,1) = reshape(capg,1,numel(capg))';
        % Rotation of the model in 2D
        th        = PolAng;
        R         = [cosd(th) -sind(th); sind(th) cosd(th)];
        capb      = capb(:,1:2)*R;
        capa      = capa(:,1:2)*R;
        capa(:,1) = round(ImSize/2)+1-capa(:,1);
        capa(:,2) = round(ImSize/2)+capa(:,2);
        capb(:,1) = round(ImSize/2)+1-capb(:,1);
        capb(:,2) = round(ImSize/2)+capb(:,2);    hold on;
        s         = 1.5;   % size of patch for image
        t         = 0:pi/5:2*pi;
        N         = length(capa);
        for i=1:N
            pb=patch(((s.*sin(t))+capa(i,1)),...
                ((s.*cos(t))+capa(i,2)),...
                'c','edgecolor','none');
            alpha(pb,.5);
            pd=patch(((s.*sin(t))+capb(i,1)),...
                ((s.*cos(t))+capb(i,2)),...
                'b','edgecolor','none');
            alpha(pd,.5);
        end
        hold off;
        %     h = plot(capa(:,1),capa(:,2),'.','color',[0 .5 1]);     set(h,'markersize',17)
        %     j = plot(capb(:,1),capb(:,2),'.','color',[.1 .1 1]);     set(j,'markersize',17);
        hold off;
        %end
        %if get(data.WProt,'value')
        hold on;
        s = 3;
        t = 0:pi/5:2*pi;
        N = length(capm);
        for i=1:N
            pb=patch(((s.*sin(t))+capm(i,1)),...
                ((s.*cos(t))+capm(i,2)),...
                'y','edgecolor','none');
            alpha(pb,.7);
        end
        hold off;
        %     hold on; h = plot(capm(:,1),capm(:,2),'.y'); hold off;
        %     set(h,'markersize',12)
    end
    % axis equal;axis tight;
    
    %set(data.eTime,'string',toc);
    %%%%%% D. Camera model
    % Down sample the image
    PS      = PixSize;               % pixel size in nm
    SI      = length(IM);
    ST      = fix(SI/PS);
    M       = IM;
    M(SI+1:PS*(ST+1),:) = 0;
    M(:,SI+1:PS*(ST+1)) = 0;
    p       = PS; 
    q       = PS;
    [m,n]   = size(M); % M is the original matrix
    M       = sum(reshape(M,p,[]),1);
    M       = reshape(M,m/p,[]).'; % Note transpose
    M       = sum(reshape(M,q,[]),1);
    M       = reshape(M,n/q,[]).'; % Note transpose
    % Noise model from electronics. Offset + Gauss electrons/sec per pixel
    % M       = M + (eDepth*.1)+(randn(size(M)).*(eNoise*(ExpTime/1000)));
    M       = M + (eDepth*.025)+((randn(size(M)).*(eNoise*(ExpTime/1000))));
    M       = floor(M./ADCval);
    % figure
    % axes(data.CamAxis);
    ma = max(M(:));
    % imagesc(M',[0 ma]); colormap gray;  axis tight
    
    %imagesc(M');colormap gray; 
    %data.M  =  M;
    % For doing all of the pixel shifts
    subshift =1;
    if subshift
        SP      = SubPix;
        PI      = floor(PS/SP);         % nm increment per image
        SX      = PI * SP;              % Search space
        M       = zeros(SI+SX,SI+SX);
        M(floor(SX/2)+1:floor(SX/2)+SI, floor(SX/2)+1:floor(SX/2)+SI)  = IM;
        x       = PS;
        y       = PS;
        MX = [];
        cnt = 0;
        for X = 1:PI:SX
            for Y = 1:PI:SX
                cnt = cnt+1;
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Changed to Ys here!!!!!!!!!
                MM = M(X:X+(ST*PS)-1,Y:Y+(ST*PS)-1);
                [m,n]   = size(MM); % M is the original matrix
               % MM = M(X:X+SI-1,Y:Y+SI-1);
                MM       = sum(reshape(MM,x,[]),1);
                MM       = reshape(MM,m/x,[]).'; % Note transpose
                MM       = sum(reshape(MM,y,[]),1);
                MM       = reshape(MM,n/y,[]).'; % Note transpose
                % Noise model from electronics
                MM       = MM + (eDepth*.1)+(randn(size(MM)).*(eNoise*(ExpTime/1000)));
                MM       = floor(MM./ADCval);
                MX(:,:,cnt) = MM;
            end
        end
        M = MX;
    end

    basepath = '/Users/aman/Desktop/FADS/polymer-tracking/images/val/data/';
    filename = strcat(basepath, 'noise_', num2str(snum));
    save(filename, 'M', 'PolAng', 'PolDiam', 'TipProb', 'TipExp', 'PhotSec', 'eNoise', 'ExpTime')

    % for drawing different subsample grids
    % for i = 1:25
    %     imagesc(M(:, :, i))
    %     pause(.5)
    % end
end



