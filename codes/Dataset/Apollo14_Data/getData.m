cd data
a = ls;
cd ..
z = a(3:6,:);
x = zeros(4,461,5);
for i=1:4 
x(i,:,:) = load(['final_Data/',z(i,:)]);
end
for i = 1:4
figure(i);
plot(x(i,:,1),x(i,:,2));
hold on;
plot(x(i,:,1),x(i,:,3));
hold on;
plot(x(i,:,1),x(i,:,4));
hold on;
plot(x(i,:,1),x(i,:,5));
end

% temp=zeros(461,5);
% i=4;
%     temp(:,:) = x(i,:,[1,2,4,6,8]);
%     save '14260Kdata.txt' 'temp' -ascii
%     
%     