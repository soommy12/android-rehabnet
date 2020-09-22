clear all

% The code is used to parse UI-PRMD dataset to the suitable data for 
% RehabNet model training which can easily compute PoseNet output.

% Note:
% Authors of the UI-PRMD dataset noticed that m03, m04, and m05
% movements are unnatural.

%% read the data

posDir = 'smallPositions';
angDir = 'smallAngles';
posTmpCsvName = 'pos.txt';
angTmpCsvName = 'ang.txt';
posFiles = dir(fullfile(posDir, '*.txt'));
angFiles = dir(fullfile(angDir, '*.txt'));
bigCsv = [];
exerciseType = 1;
for k = 1:length(posFiles)
    if k/10 > exerciseType
        exerciseType = exerciseType + 1;
    end
    currentPosFile = posFiles(k).name;
    currentAngFile = angFiles(k).name;
    fullPosFileName = fullfile(posDir, currentPosFile);
    fullAngFileName = fullfile(angDir, currentAngFile);
    fprintf(1, 'Now reading %s\n', fullPosFileName);
%     tf = strcmp(currentPosFile, 'm01_s01_positions.txt');
%     if tf == 1
%         continue
%     end
    dataPos = load(fullPosFileName);
    dataAng = load(fullAngFileName);
    csvwrite(posTmpCsvName, dataPos);
    csvwrite(angTmpCsvName, dataAng);
    readpos = csvread(posTmpCsvName);
    readang = csvread(angTmpCsvName);
    % smooth the data
    % smooth the joint angles more because the measurements are noisier
    readpos_sm = zeros(size(readpos,1),size(readpos,2));
    readang_sm = zeros(size(readpos,1),size(readpos,2));
    for i=1:size(readpos,2)
        readpos_sm(:,i) = smooth(readpos(:,i),5);
        readang_sm(:,i) = smooth(readang(:,i),20,'rloess');
    end

    % swap the columns and rows
    readpos1 = readpos_sm';
    readang1 = readang_sm';

    % number of time frames
    num_frames = size(readpos,1);

    % reshape data into form [22 x 3 x number_frames]
    % skeleton = zeros(22,3,num_frames);
    skeleton = zeros(22,3,num_frames);

    for i=1:num_frames
        skeleton(:,:,i) =reshape(readpos1(:,i),[3,22])';
    end
    skeleton_ang = zeros(22,3,num_frames);
    for i=1:num_frames
        skeleton_ang(:,:,i) =reshape(readang1(:,i),[3,22])';
    end

    %% order of joint connections
    J = [4, 6, 5, 3, 2, 3, 7, 8, 9, 3, 11,  12, 13, 1,  15, 16, 17, 1,  19, 20, 21;
         3, 5, 3, 2, 1, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];

    % 1 Waist (absolute)
    % 2 Spine
    % 3 Chest
    % 4 Neck
    % 5 Head
    % 6 Head tip
    % 7 Left collar
    % 8 Left upper arm 
    % 9 Left forearm
    % 10 Left hand
    % 11 Right collar
    % 12 Right upper arm 
    % 13 Right forearm
    % 14 Right hand
    % 15 Left upper leg 
    % 16 Left lower leg 
    % 17 Left foot 
    % 18 Left leg toes
    % 19 Right upper leg 
    % 20 Right lower leg 
    % 21 Right foot
    % 22 Right leg toes

    %% find maximum and minimum values (for plotting)
    ss = [];
    for i = 1:num_frames
        ss = [ss; skeleton(:,:,i)];
    end

    maxx = max(ss(:,1));
    minx = min(ss(:,1));
    maxy = max(ss(:,2));
    miny = min(ss(:,2));
    maxz = max(ss(:,3));
    minz = min(ss(:,3));

    clear ss

    %% convert the data from relative coordinates to absolute coordinates

    % Coordinate system:
    % +X is going screen right 
    % +Y is going up
    % +Z is coming out of the screen 

    % skel = zeros(22,3,num_frames);
    skel = zeros(12,2,num_frames);
    for i=1:num_frames

        joint = skeleton(:,:,i);
        joint_ang =  skeleton_ang(:,:,i);

        % chest, neck, head
        rot_2 = eulers_2_rot_matrix(joint_ang(1,:)*pi/180);
        joint(2,:) =  (rot_2*joint(2,:)')' +  joint(1,:);
        rot_3 = rot_2*eulers_2_rot_matrix(joint_ang(2,:)*pi/180);
        joint(3,:) =  (rot_3*joint(3,:)')' +  joint(2,:);
        rot_4 = rot_3*eulers_2_rot_matrix(joint_ang(3,:)*pi/180);
        joint(4,:) =  (rot_4*joint(4,:)')' +  joint(3,:);
        rot_5 = rot_4*eulers_2_rot_matrix(joint_ang(4,:)*pi/180);
        joint(5,:) =  (rot_5*joint(5,:)')' +  joint(4,:);
        rot_6 = rot_5*eulers_2_rot_matrix(joint_ang(5,:)*pi/180);
        joint(6,:) =  (rot_6*joint(6,:)')' +  joint(5,:);

        % left arm
        rot_7 = eulers_2_rot_matrix(joint_ang(3,:)*pi/180);
        joint(7,:) =  (rot_7*joint(7,:)')' +  joint(3,:);
        rot_8 = rot_7*eulers_2_rot_matrix(joint_ang(7,:)*pi/180);
        joint(8,:) =  (rot_8*joint(8,:)')' +  joint(7,:);
        rot_9 = rot_8*eulers_2_rot_matrix(joint_ang(8,:)*pi/180);
        joint(9,:) =  (rot_9*joint(9,:)')' +  joint(8,:);
        rot_10 = rot_9*eulers_2_rot_matrix(joint_ang(9,:)*pi/180);
        joint(10,:) = (rot_10*joint(10,:)')' +  joint(9,:);

        % right arm
        rot_11 = eulers_2_rot_matrix(joint_ang(3,:)*pi/180);
        joint(11,:) =  (rot_11*joint(11,:)')' +  joint(3,:);
        rot_12 = rot_11*eulers_2_rot_matrix(joint_ang(11,:)*pi/180);
        joint(12,:) =  (rot_12*joint(12,:)')' +  joint(11,:);
        rot_13 = rot_12*eulers_2_rot_matrix(joint_ang(12,:)*pi/180);
        joint(13,:) =  (rot_13*joint(13,:)')' +  joint(12,:);
        rot_14 = rot_13*eulers_2_rot_matrix(joint_ang(13,:)*pi/180);
        joint(14,:) =  (rot_14*joint(14,:)')' +  joint(13,:);

        % left leg
        rot_15 = eulers_2_rot_matrix(joint_ang(1,:)*pi/180);
        joint(15,:) =  (rot_15*joint(15,:)')' +  joint(1,:);
        rot_16 = rot_15*eulers_2_rot_matrix(joint_ang(15,:)*pi/180);
        joint(16,:) =  (rot_16*joint(16,:)')' +  joint(15,:);
        rot_17 = rot_16*eulers_2_rot_matrix(joint_ang(16,:)*pi/180);
        joint(17,:) =  (rot_17*joint(17,:)')' +  joint(16,:);
        rot_18 = rot_17*eulers_2_rot_matrix(joint_ang(17,:)*pi/180);
        joint(18,:) =  (rot_18*joint(18,:)')' +  joint(17,:);

        % right leg
        rot_19 = eulers_2_rot_matrix(joint_ang(1,:)*pi/180);
        joint(19,:) =  (rot_19*joint(19,:)')' +  joint(1,:);
        rot_20 = rot_19*eulers_2_rot_matrix(joint_ang(19,:)*pi/180);
        joint(20,:) =  (rot_20*joint(20,:)')' +  joint(19,:);
        rot_21 = rot_20*eulers_2_rot_matrix(joint_ang(20,:)*pi/180);
        joint(21,:) =  (rot_21*joint(21,:)')' +  joint(20,:);
        rot_22 = rot_21*eulers_2_rot_matrix(joint_ang(21,:)*pi/180);
        joint(22,:) =  (rot_22*joint(22,:)')' +  joint(21,:);

        % remove Z 
        joint(:,3) = [];
        % remove unnecessary joints
        joint([1 2 3 4 5 6 7 11 18,22],:) = [];

        % define image size 
        imgSize = 100;
        % calculate scale 
        scale = max(maxx - minx, maxy - miny);

        joint(:,1) = joint(:,1) - ((maxx + minx)/2);
        joint(:,2) = joint(:,2) - ((maxy + miny)/2);

        joint(:,:) = joint(:,:) / scale;

        joint(:,1) = (joint(:,1) + 0.5)*imgSize;
        joint(:,2) = (joint(:,2) + 0.5)*imgSize;

        joint(:,1) = interp1([1 max(joint(:,1),[],1)], [1, 28], joint(:,1));
        joint(:,2) = interp1([1 max(joint(:,2),[],1)], [1, 28], joint(:,2));

        % if value is out of the scaled scope just put 0 there
        joint(isnan(joint)) = 0;

        skel(:,:,i) = joint;

        rowJoint = joint(:)';  % this is to get one row instead of matrix
        rowJointWithType = [rowJoint exerciseType];
        
        bigCsv = [bigCsv;rowJointWithType];
    end
    disp('clearing...')
    
    delete(posTmpCsvName)
    delete(angTmpCsvName)
    clearvars readpos readang
end

 %% save data to csv
% cHeader = {'leftUpperArmX' 'leftForearmX' 'leftHandX' 'rightUpperArmX' 'rightForearmX' 'rightHandX' 'leftUpeerLegX' 'leftLowerLegX' 'leftFootX' 'rightUpperLegX' 'rightLoweLegX' 'rightFootX' 'leftUpperArmY' 'leftForearmY' 'leftHandY' 'rightUpperArmY' 'rightForearmY' 'rightHandY' 'leftUpeerLegY' 'leftLowerLegY' 'leftFootY' 'rightUpperLegY' 'rightLoweLegY' 'rightFootY' 'Type' }; % labels
cHeader = {num2str(size(bigCsv,1)), num2str(24), 'deep_squat' 'hurdle_step' 'side_lunge' 'sit_to_stand' 'standing_active_straight_leg_raise' 'standing_shoulder_abduction' 'standing_shoulder_extension' 'standing_shoulder_internal_external_rotation' 'standing_shoulder_scaption' };
commaHeader = [cHeader;repmat({','},1,numel(cHeader))];
commaHeader = commaHeader(:)';
textHeader = cell2mat(commaHeader);
%write labels to file
fid = fopen('rehab_train.csv','w'); 
fprintf(fid,'%s\n',textHeader)
fclose(fid)
%write data to end of file
dlmwrite('rehab_train.csv',bigCsv,'-append');
display('FINISHED!');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Functions

function [ R ] = eulers_2_rot_matrix(x)
    %EULER_2_ROT_MATRIX transforms a set of euler angles into a rotation  matrix 
    % input vector of euler angles 
    % [gamma_x, beta_y, alpha_z]  are ZYX Eulers angles in radians
    gamma_x=x(1);beta_y=x(2);alpha_z=x(3);
    R = rotz(alpha_z) * roty(beta_y) * rotx(gamma_x);
end

function r = rotx(t)
    % ROTX Rotation about X axis
    ct = cos(t);
    st = sin(t);
    r =    [1	0	0
            0	ct	-st
            0	st	ct];
end

function r = roty(t)
    % ROTY Rotation about Y axis
    ct = cos(t);
    st = sin(t);
    r =    [ct	0	st
            0	1	0
            -st	0	ct];
end

function r = rotz(t)
    %ROTZ Rotation about Z axis
	ct = cos(t);
	st = sin(t);
	r =    [ct	-st	0
		st	ct	0
		0	0	1];
end