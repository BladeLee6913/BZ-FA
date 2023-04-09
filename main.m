close all;
clear;
clc;

tic;
%% loading data
points_angle = 0: 2*pi/250: 2*pi - 2*pi/250;
load('Amplitudes.mat');
load('height.mat');
load('ToF.mat');

T_raw = -T_ini+255;
T_left = eye(100);
T_right = eye(250);

%% initiation
T=300;
N=20;
dim =1;
rng(1);
X1_Min = 1;
X1_Max = 100;
X2_Min = 1;
X2_Max = 250;

%initial
vec_flag=[1,-1];
Threshold=0.192;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;

Current_X1 = round(X1_Min+rand(N,1)*(X1_Max-X1_Min));
Current_X2 = round(X2_Min+rand(N,1)*(X2_Max-X2_Min));

for i = 1:N
    right = [T_right(:, Current_X2(i, 1)+1:end), T_right(:, 1:Current_X2(i, 1))];
    left=  [T_left( Current_X1(i,1)+1:end, :); T_left(1: Current_X1(i,1), :)];
    ytest = left*A*right;
    L2values(i) = L2error(normalize(T_raw, 'range'), normalize(ytest, 'range'));
end


%
[Current_X_Fitness, indexes] = min(L2values);
X1food = Current_X1(indexes,:);
X2food = Current_X2(indexes,:);


Nm=round(N/2);
Nf=N-Nm;

%
X1m = Current_X1(1:Nm,:);
X1f =  Current_X1(Nm+1:N,:);
X2m = Current_X2(1:Nm,:);
X2f =  Current_X2(Nm+1:N,:);


%
L2values_m = L2values(1:Nm);
L2values_f = L2values(Nm+1:N);

%
[BestL2values_m, gindexes1] = min(L2values_m);
X1best_m = X1m(gindexes1, :);
X2best_m = X2m(gindexes1, :);
[BestL2values_f, gindexes2] = min(L2values_f);
X1best_f = X1f(gindexes2, :);
X2best_f = X2f(gindexes2, :);

%% iteration
for t = 1:T
    Temp=exp(-((t)/T));  
    Q=C1*exp(((t-T)/(T)));
    if Q>1        Q=1;    end
    % Exploration Phase 1
    if Q<Threshold
        for i=1:Nm
            for j=1:1:dim

                rand_leader_index1 = floor(Nm*rand()+1);
                rand_leader_index2 = floor(Nm*rand()+1);
                X1_randm = X1m(rand_leader_index1, :);
                X2_randm = X2m(rand_leader_index2, :);
                flag_index1 = floor(2*rand()+1);
                Flag1=vec_flag(flag_index1);
                Am=exp(-L2values_m(rand_leader_index1)/(L2values_m(i)+eps));
                X1newm(i,j)=X1_randm(j)+Flag1*C2*Am*((X1_Max-X1_Min)*rand+X1_Min);
                X2newm(i,j)=X2_randm(j)+Flag1*C2*Am*((X2_Max-X2_Min)*rand+X2_Min);

            end
        end
        for i=1:Nf
            for j=1:1:dim

                %
                rand_leader_index1 = floor(Nf*rand()+1);
                rand_leader_index2 = floor(Nf*rand()+1);
                X1_randf = X1f(rand_leader_index1, :);
                X2_randf = X2f(rand_leader_index2, :);
                flag_index2 = floor(2*rand()+1);
                Flag2=vec_flag(flag_index2);
                Af=exp(-L2values_f(rand_leader_index1)/(L2values_f(i)+eps));
                X1newf(i,j)=X1_randf(j)+Flag2*C2*Af*((X1_Max-X1_Min)*rand+X1_Min);
                X2newf(i,j)=X2_randf(j)+Flag2*C2*Af*((X2_Max-X2_Min)*rand+X2_Min);

            end
        end
    else %Exploitation Phase 2
        if Temp>Thresold2  
            for i=1:Nm
                %
                flag_index1 = floor(2*rand()+1);
                Flag1=vec_flag(flag_index1);
                for j=1:1:dim
                    X1newm(i,j)=X1food(j)+C3*Flag1*Temp*rand*(X1food(j)-X1m(i,j));
                end
                flag_index2 = floor(2*rand()+1);
                Flag2=vec_flag(flag_index1);
                for j=1:1:dim
                    X2newm(i,j)=X2food(j)+C3*Flag1*Temp*rand*(X2food(j)-X2m(i,j));
                end

            end
            for i=1:Nf
                %
                flag_index2 = floor(2*rand()+1);
                Flag2=vec_flag(flag_index2);
                for j=1:1:dim
                    X1newf(i,j)=X1food(j)+Flag2*C3*Temp*rand*(X1food(j)-X1f(i,j));
                end
                flag_index2 = floor(2*rand()+1);
                Flag2=vec_flag(flag_index2);
                for j=1:1:dim
                    X2newf(i,j)=X2food(j)+Flag2*C3*Temp*rand*(X2food(j)-X2f(i,j));
                end


            end
        else 
            if rand>0.6 
                for i=1:Nm
                    for j=1:1:dim
                      

                        %
                        FM1=exp(-(BestL2values_m)/(L2values_m(i)+eps));
                        X1newm(i,j) = X1m(i,j) +C3*FM1*rand*(Q*X1best_f(j)-X1m(i,j));
                        X2newm(i,j) = X2m(i,j) +C3*FM1*rand*(Q*X2best_f(j)-X2m(i,j));

                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                        
                        %
                        FF1=exp(-(BestL2values_f)/(L2values_f(i)+eps));%eq.(13)
                        X1newf(i,j) = X1f(i,j) +C3*FF1*rand*(Q*X1best_m(j)-X1f(i,j));
                        X2newf(i,j) = X2f(i,j) +C3*FF1*rand*(Q*X2best_m(j)-X2f(i,j));

                    end
                end
            else
                for i=1:Nm
                    for j=1:1:dim
                      
                        
                        X1Mm=exp(-L2values_f(i)/(L2values_m(i)+eps));%eq.(17)
                        X1newm(i,j)=X1m(i,j) +C3*rand*X1Mm*(Q*X1f(i,j)-X1m(i,j));%eq.(15
                        X2Mm=exp(-L2values_f(i)/(L2values_m(i)+eps));%eq.(17)
                        X1newm(i,j)=X2m(i,j) +C3*rand*X2Mm*(Q*X2f(i,j)-X2m(i,j));%eq.(15

                    end
                end
                for i=1:Nf
                    for j=1:1:dim
                       
                        
                        X1Mf = exp(-L2values_m(i)/(L2values_f(i)+eps));
                        X1newf(i,j) = X1f(i,j) +C3*rand*X1Mf*(Q*X1m(i,j)-X1f(i,j));%eq.(16)
                        X2Mf = exp(-L2values_m(i)/(L2values_f(i)+eps));
                        X1newf(i,j) = X2f(i,j) +C3*rand*X2Mf*(Q*X2m(i,j)-X2f(i,j));%eq.(16)

                    end
                end
                flag_index = floor(2*rand()+1);
                egg=vec_flag(flag_index);
                if egg==1             
                    [GYworst1, gworst1] = max(L2values_m);
                    X1newm(gworst1,:)=round(X1_Min+rand*(X1_Max-X1_Min));%eq.(19)
                    X2newm(gworst1,:)=round(X2_Min+rand*(X2_Max-X2_Min));%eq.(19)
                    [GYworst2, gworst2] = max(L2values_f);
                    X1newf(gworst2,:)=round(X1_Min+rand*(X1_Max-X1_Min)); %eq.(19)
                    X2newf(gworst2,:)=round(X2_Min+rand*(X2_Max-X2_Min)); %eq.(19)
                end
            end
        end
    end



    for j=1:Nm
     
        %
        X1Flag4ub=X1newm(j,:)>X1_Max;
        X1Flag4lb=X1newm(j,:)<X1_Min;
        X2Flag4ub=X2newm(j,:)>X2_Max;
        X2Flag4lb=X2newm(j,:)<X2_Min;
        X1newm(j,:)=round((X1newm(j,:).*(~(X1Flag4ub+X1Flag4lb)))+X1_Max.*X1Flag4ub+X1_Min.*X1Flag4lb);
        X2newm(j,:)=round((X2newm(j,:).*(~(X2Flag4ub+X2Flag4lb)))+X2_Max.*X2Flag4ub+X2_Min.*X2Flag4lb);

        right = [T_right(:, X2newm(j, 1)+1:end), T_right(:, 1:X2newm(j, 1))];
        left=  [T_left( X1newm(j,1)+1:end, :); T_left(1: X1newm(j,1), :)];
        ytest = left*A*right;
        y = L2error(normalize(T_raw, 'range'), normalize(ytest, 'range'));
        if y<L2values_m(j)
            L2values_m(j) = y;
            X1m(j,:) = X1newm(j,:);
            X2m(j,:) = X2newm(j,:);
        end
    end

    %
    [L2minbest1, gindexbest1] = min(L2values_m);


    for j=1:Nf
                
        %
        X1Flag4ub=X1newf(j,:)>X1_Max;
        X1Flag4lb=X1newf(j,:)<X1_Min;
        X2Flag4ub=X2newf(j,:)>X2_Max;
        X2Flag4lb=X2newf(j,:)<X2_Min;
        X1newf(j,:)=round((X1newf(j,:).*(~(X1Flag4ub+X1Flag4lb)))+X1_Max.*X1Flag4ub+X1_Min.*X1Flag4lb);
        X2newf(j,:)=round((X2newf(j,:).*(~(X2Flag4ub+X2Flag4lb)))+X2_Max.*X2Flag4ub+X2_Min.*X2Flag4lb);

        right = [T_right(:, X2newf(j, 1)+1:end), T_right(:, 1:X2newf(j, 1))];
        left=  [T_left( X1newf(j,1)+1:end, :); T_left(1: X1newf(j,1), :)];
        ytest = left*A*right;
        y = L2error(normalize(T_raw, 'range'), normalize(ytest, 'range'));
        if y<L2values_f(j)
            L2values_f(j) = y;
            X1f(j,:) = X1newf(j,:);
            X2f(j,:) = X2newf(j,:);
        end
    end

    %
    [L2minbest2, gindexbest2] = min(L2values_f);

    %
    if L2minbest1<BestL2values_m
        X1best_m = X1m(gindexbest1,:);
        X2best_m = X2m(gindexbest1,:);
        BestL2values_m = L2minbest1;
    end

    %
    if L2minbest2<BestL2values_f
        X1best_f = X1f(gindexbest2,:);
        X2best_f = X2f(gindexbest2,:);
        BestL2values_f = L2minbest2;
    end

    %
    if L2minbest1<L2minbest2
        L2gbest_t(t) = min(L2minbest1);
    else
        L2gbest_t(t) = min(L2minbest2);
    end

    %
    if BestL2values_m<BestL2values_f
        L2GYbest = BestL2values_m;
        X1food = X1best_m;
        X2food = X2best_m;
    else
        L2GYbest = BestL2values_f;
        X1food = X1best_f;
        X2food = X2best_f;

    end
end

MinFitness = L2GYbest;
A_transmutation = left*A*right;

toc;
%% drawing
figure;
plot(L2gbest_t');
title('L2 error');

figure;
subplot(131);
imagesc(T_raw);
title('Time images');
colorbar;
subplot(132);
imagesc(A);
title('Amplitude images');
colorbar;
subplot(133);
imagesc(A_transmutation);
title('After registration')
colorbar;

point_x = T_ini.*cos(points_angle);
point_y = T_ini.*sin(points_angle);
point_z = repmat(H, 1, 250);

X_shape = reshape(point_x', [],1);
Y_shape = reshape(point_y', [],1);
Z_shape = reshape(point_z', [],1);

amp = reshape(A_transmutation',[],1);

fig = figure('Name', 'Point cloud');
fig.Position = [180  58   860   980];
scatter3(X_shape, Y_shape, Z_shape, 10, amp, 'filled');
% colorbar;
set(gca, 'linewidth', 2.1, 'fontsize', 20, 'fontname', 'times') ;


