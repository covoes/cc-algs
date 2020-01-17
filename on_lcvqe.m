function [centroids partition SSQ time] = on_lcvqe(X, initCenters, learnRate, delearnRate, constraints, procOrder,epoch)
%ON_LCVQE Implements  an online version of the LCVQE algorithm described in: T. F. Covoes, E. R. Hruschka and J. Ghosh, "Competitive Learning With Pairwise Constraints," in IEEE Transactions on Neural Networks and Learning Systems, vol. 24, no. 1, pp. 164-169, Jan. 2013.
% X: numObjects x numFeatures matrix
% initCenter: the initial prototypes numClusters x numFeatures matrix
% learnRate: rate for attraction between the prototype and a vector
% delearnRate: rate for a prototype moving away from a vector
% constraints: ML and CL constraints numConstraints x 3 matrix
% procOrder: processing order to simulate online setting, 1 x numObjects matrix

numObjects =size(X,1);
centroids=initCenters;
t=tic;
for p=1:length(procOrder)
	i = procOrder(p);
	distancesXi = sum( bsxfun(@minus, X(i,:), centroids) .^ 2, 2);
    [ ~, j ] = min(distancesXi);

	constraintsInvolved = find( (constraints(:,1) == i) | (constraints(:,2) == i ) );

	if isempty(constraintsInvolved)
		centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
	else
		for l=constraintsInvolved'
			otherObject = constraints(l,1);
			if(otherObject == i)
				otherObject = constraints(l,2);
			end

			%only deals with constraints regarding objects already processed
			if epoch==1 && isempty(find(procOrder(1:p) == otherObject,1))
				continue;
			end


			%find the cluster closest to the other object
			distancesXo = sum( bsxfun(@minus, X(otherObject,:), centroids) .^ 2, 2);
		    [ ~, r ] = min(distancesXo);

			typeConstraint = constraints(l,3);
			if typeConstraint == 1
				centroids = lcMLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo);
			else
				centroids = lcCLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo);
			end

		end
	end
end
time=toc(t);
%	f=figure;
%	plot(X(:,1),X(:,2),'b.');
%	hold all;
%	plot(centroids(:,1),centroids(:,2),'rx', 'MarkerSize', 20);
%	plot(initCenters(:,1),initCenters(:,2),'kd', 'MarkerSize', 20);
%	print(f, '-depsc2', sprintf('figs/clust_p.eps'));
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
