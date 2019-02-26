cd data
a = ls;
cd ..
z = a(3:8,:);
x = zeros(6,461,5);
for i=1:6
x(i,:,:) = load(['final_Data/',z(i,:)]);
end
for i = 1:6
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
% i=6;
%     temp(:,:) = x(i,:,[1,2,4,6,8]);
%     temp_str = ['final_Data/',z(i,:)];
%     save '67481Kdata.txt' 'temp' -ascii
