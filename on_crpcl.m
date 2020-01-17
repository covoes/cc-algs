function [centroids partition SSQ time w] = on_crpcl(X, initCenters, learnRate, delearnRate, constraints, procOrder, epoch, w, computeSSQ, constList)
%ON_CRPCL Implements  the Constrained Rival Penalized Competitive Learning algorithm described in: T. F. Covoes, E. R. Hruschka and J. Ghosh, "Competitive Learning With Pairwise Constraints," in IEEE Transactions on Neural Networks and Learning Systems, vol. 24, no. 1, pp. 164-169, Jan. 2013.
% X: numObjects x numFeatures matrix
% initCenter: the initial prototypes numClusters x numFeatures matrix
% learnRate: rate for attraction between the prototype and a vector
% delearnRate: rate for a prototype moving away from a vector
% constraints: ML and CL constraints numConstraints x 3 matrix
% procOrder: processing order to simulate online setting, 1 x numObjects matrix

assert(learnRate > 0)
assert( delearnRate > 0 && delearnRate < learnRate)
assert(~isempty(procOrder))
assert(epoch > 0)

numObjects=size(X,1);
k=size(initCenters,1);
centroids=initCenters;
if nargin < 7
	w = ones([k 1]);
end
if nargin == 7
	computeSSQ = 1;
end

gamma=ones([k 1])*(1/k);
t=tic;
for p=1:length(procOrder)
	i = procOrder(p);
	distancesXi = sum( bsxfun(@minus, X(i,:), centroids) .^ 2, 2) .* gamma ;
    [ ~, j ] = min(distancesXi);

	%constraintsInvolved = find( (constraints(:,1) == i) | (constraints(:,2) == i ) );
	constraintsInvolved = constList{i};

	if isempty(constraintsInvolved)
		centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
		w(j) = w(j) + 1;
		distancesXi(j) = NaN;
		[ ~, n ] = min(distancesXi);
		centroids(n,:) = centroids(n,:) - delearnRate * (X(i,:) - centroids(n,:));
	else
		forbidClusters = [];
		for l=constraintsInvolved'
			otherObject = constraints(l,1);
			if(otherObject == i)
				otherObject = constraints(l,2);
			end

			%only deals with constraints regarding objects already processed
			if epoch == 1 && isempty(find(procOrder(1:p) == otherObject,1))
				continue;
			end


			%find the cluster closest to the other object
			distancesXo = sum( bsxfun(@minus, X(otherObject,:), centroids) .^ 2, 2).*gamma;
		    [ ~, r ] = min(distancesXo);
			forbidClusters = [ forbidClusters; r];
		end
		forbidClusters = unique(forbidClusters);
		tmp=distancesXi;
		tmp(j)=NaN;
		[~, n] = min(tmp);

		%if there is no violation assigning to the cluster or no other option, do it
		if ((isempty(find(ismember(j,forbidClusters),1))) || (length(forbidClusters) == k))
			%RPCL
			centroids(n,:) = centroids(n,:) - delearnRate * (X(i,:) - centroids(n,:));
			centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
			w(j) = w(j) + 1;
		else
			%otherwise just put the object on the nearest valid centroid
			[~, idxSorted]=sort(distancesXi,'ascend');

			valid=find(~ismember(idxSorted,forbidClusters));
			n=idxSorted(valid(1));
				
			%attract neighbor
			centroids(n,:) = centroids(n,:) + learnRate * (X(i,:) - centroids(n,:));
			centroids(j,:) = centroids(j,:) - delearnRate * (X(i,:) - centroids(j,:));
			w(n) = w(n) + 1;
		end
	end

	gamma = w / sum(w);

end
time=toc(t);
%	f=figure;
%	plot(X(:,1),X(:,2),'b.');
%	hold all;
%	plot(centroids(:,1),centroids(:,2),'rx', 'MarkerSize', 10);
%	plot(initCenters(:,1),initCenters(:,2),'kd', 'MarkerSize', 10);
%	print(f, '-depsc2', sprintf('figs/clust_p.eps'));
%	pause
%	close(f);

partition = zeros([1 numObjects]);
if computeSSQ
	SSQ = 0;
	for i=1:numObjects
		distancesXi = sum( bsxfun(@minus, X(i,:), centroids) .^ 2, 2);
		[ minDist j ] = min(distancesXi);
		partition(i) = j;
		SSQ = SSQ + minDist;
	end
else
	SSQ = -1;
end

end
