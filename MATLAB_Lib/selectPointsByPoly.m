function [ isIn ] = selectPointsByPoly( points )
    %UNTITLED �˴���ʾ�йش˺�����ժҪ
    %   �˴���ʾ��ϸ˵��
    figure;
    scatter(points(:,1),points(:,2),10,'filled');
    pxy = impoly();
    pxy = pxy.getPosition();
    isIn = inpolygon(points(:,1),points(:,2),pxy(:,1),pxy(:,2)); 
    hold on;
    scatter(points(isIn,1),points(isIn,2),10,'filled');
end

