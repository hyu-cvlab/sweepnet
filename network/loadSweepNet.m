function [net1, net2] = loadSweepNet(path, varargin)
load(path);
net1 = dagnn.DagNN.loadobj(net_obj1);
net2 = dagnn.DagNN.loadobj(net_obj2);
net1.move('gpu');
net2.move('gpu');
net1.mode = 'test';
net2.mode = 'test';
end