function [centroids partition SSQ time w] = on_coopcrpcl(X, initCenters, learnRate, constraints, procOrder, epoch, w)
%ON_COOPCRPCL Implements  an constrained version of the online algorithm CPCL (ICANN, 2010)
% X: numObjects x numFeatures data matrix
% initCenter: the initial prototypes numClusters x numFeatures matrix
% learnRate: rate for attraction between the prototype and a vector
% constraints: ML and CL constraints numConstraints x 3 matrix
% procOrder: processing order to simulate online setting, 1 x numObjects matrix

numObjects=size(X,1);
k=size(initCenters,1);
centroids=initCenters;
t=tic;

%used to control the cooperation
if nargin < 6
	epoch = 1;
	w=ones([k 1]);
end
gamma = w / sum(w);

for p=1:length(procOrder)
	i = procOrder(p);

	%initially, find out who could be penalized according to CL
	constraintsInvolved = find( (constraints(:,1) == i) | (constraints(:,2) == i ) );

	couldBePenalized = [];
	for l=constraintsInvolved'
		otherObject = constraints(l,1);
		if(otherObject == i)
			otherObject = constraints(l,2);
		end

		%only deals with constraints regarding objects already processed in the first epoch
		if epoch == 1 && isempty(find(procOrder(1:p) == otherObject,1))
			continue;
		end

		%find the cluster closest to the other object
		distancesXo = sum( bsxfun(@minus, X(otherObject,:), centroids) .^ 2, 2);
	    [ ~, r ] = min(distancesXo);
		typeConstraint = constraints(l,3);
		%not handling ml constraints by now
		if typeConstraint == -1
			couldBePenalized = [ couldBePenalized r ];
		end
	end

	distancesXiNorm = sum( bsxfun(@minus, X(i,:), centroids) .^ 2, 2);
	distancesXi = distancesXiNorm .* gamma;
	[~, idxSortedXi ] = sort(distancesXi);

	%ideally, the winner is the closest cluster, but CL can make it different
	if isempty(couldBePenalized)
		winner = idxSortedXi(1);
	else
		currentWinner = 0;
		while true 
			currentWinner = currentWinner + 1;
			if currentWinner > k
				%if every cluster violates contraints, go to the closest one
				winner = idxSortedXi(1);
				break;
			end
			winner = idxSortedXi(currentWinner);
			if ~ismember(couldBePenalized, winner)
				break;
			end
		end
	end
	w(winner) = w(winner) + 1;

	distanceXiW = sqrt(distancesXiNorm(winner));

	distCents = squareform(pdist(centroids));
	neighborhoodWinner = find(distCents(winner,:) <= distanceXiW);

	[~, idxSortNeighborhood] = sort(distCents(winner,neighborhoodWinner));

	idxSortNeighborhood = neighborhoodWinner(idxSortNeighborhood);

	penalizedSet = intersect( idxSortNeighborhood, couldBePenalized );
	%difference between vectors without the sort performed by setdiff
	mmb = ismember(idxSortNeighborhood, penalizedSet );
	coopSet = idxSortNeighborhood(~mmb);

	q = min(length(coopSet), epoch);
	coop = coopSet(1:q);

	penalizedSet = [penalizedSet coopSet(q+1:end)];


	%update cooperators
	for c=coop
		distXiCoop = sqrt(sum((centroids(c,:)-X(i,:)).^2));
		distProport = distanceXiW/max(distanceXiW,distXiCoop);
		centroids(c,:) = centroids(c,:) + learnRate * distProport * (X(i,:) - centroids(c,:));
	end

	%update penalized
	for pe=penalizedSet
		distXiPen = sqrt(sum((centroids(pe,:)-X(i,:)).^2));
		distProport = distanceXiW/distXiPen;
		centroids(pe,:) = centroids(pe,:) - learnRate * distProport * (X(i,:) - centroids(pe,:));
	end

	gamma = w / sum(w);

end
time=toc(t);


%do not eliminate clusters for now
%dists = squareform(pdist(centroids));
%idxDel=[];
%for j=1:k
%	tmp = find( dists(j,:) < 1e-2 );
%	tmp = setdiff(tmp,1:j);
%	idxDel = [ idxDel tmp ];
%end
%idxDel = unique(idxDel);
%idxOk = setdiff(1:k,idxDel);
%centroids = centroids(idxOk,:);
%w = w(idxOk);
%k = k - length(idxDel);

%fprintf('epoch %d deleting %d\n',epoch,length(idxDel));
%
%
%	f=figure;
%	plot(X(:,1),X(:,2),'b.');
%	hold all;
%	plot(centroids(:,1),centroids(:,2),'rx', 'MarkerSize', 20);
%	plot(initCenters(:,1),initCenters(:,2),'kd', 'MarkerSize', 20);
%	print(f, '-dpng', sprintf('clust_p%03d.png',epoch));
%	pause
%	close(f);




partition = zeros([1 numObjects]);
SSQ = 0;
for i=1:numObjects
	distancesXi = sum( bsxfun(@minus, X(i,:), centroids) .^ 2, 2);
    [ minDist j ] = min(distancesXi);
	partition(i) = j;
	SSQ = SSQ + minDist;
end

end
