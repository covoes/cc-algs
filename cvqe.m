function [idx centroids iter CVQE time] = cvqe(X, k, constraints)
%CVQE Constrained vector quantization error (Davidson & Ravi, 2005)
% Modification on the k-means algorithm to penalize solutions that 
% violates pairwise constraints
% conList is a Cx3 matrix with the first and second values refering to 
% objects and the third column equal to 1 or -1, if the constraint
% is must-link or cannot-link
time=NaN;
tstart = tic;
numObjects = size(X,1);

seeds = randsample( numObjects, k );
centroids = X(seeds, :);

distances = zeros( numObjects, k );
maxIter = 500;
iter = 1;
oldIdx = [];
idx = 1;

MLs = find( constraints(:,3) == 1 )';
CLs = find( constraints(:,3) == -1)';

while iter<=maxIter && ~isequal(oldIdx, idx) 
	%set of itens violating must and cannot link constraints
	GMLV = repmat(struct( 'item', [], 'cost', [] ), k, 1);
	GCLV = repmat(struct( 'item', [], 'cost', [] ), k, 1);

	oldIdx = idx;

	for c=1:k
		diffToCentroid = X - repmat(centroids(c,:), numObjects, 1);
		distances(:,c) = sum( diffToCentroid .^ 2, 2 ); 
	end
	[ minDists idx ] = min(distances, [], 2);

	%need to compute the distance between all centroids
	%the inf is used so that the distance between each cluster and itself stay larger than everything
	%this is used to facilitate the search for the nearest centroid 
	distanceBetweenCentroids = inf*ones(k);
	for k1=1:k
		for k2=k1+1:k
			distanceBetweenCentroids(k1,k2) = sum((centroids(k1,:) - centroids(k2,:)).^2);
			distanceBetweenCentroids(k2,k1) = distanceBetweenCentroids(k1,k2);
		end
	end

	%must link treatment
	for c=MLs
		s_1 = constraints(c, 1);
		s_2 = constraints(c, 2);
		c_j = idx(s_1);
		c_n = idx(s_2);

		%constraint already being satisfied
		if c_j == c_n
			continue;
		end


		%need to calculate all the possible assignments
		%and choose the one that causes the smallest CVQE
		newClusterForS_1 = 0;		
		newClusterForS_2 = 0;		
		smallerCost = inf;
		for k1=1:k
			for k2=k1:k
				if k1 ~= k2
					%for the cases of violation of constraints calculates the cost involved
					costInitial = 0.5*distanceBetweenCentroids(k1,k2);
				else
					costInitial = 0;
				end

				%check the cost of putting s_1 in k1 and s_2 in k2
				cost = costInitial + 0.5*distances(s_1,k1) + 0.5*distances(s_2,k2);
				if(cost < smallerCost)
					newClusterForS_1 = k1;
					newClusterForS_2 = k2;
					smallerCost = cost;
				end
				%check the cost of putting s_1 in k2 and s_2 in k1
				cost = costInitial + 0.5*distances(s_1,k2) + 0.5*distances(s_2,k1);
				if(cost < smallerCost)
					newClusterForS_1 = k2;
					newClusterForS_2 = k1;
					smallerCost = cost;
				end
			end
		end
		idx(s_1) = newClusterForS_1;
		idx(s_2) = newClusterForS_2;

		%if constraints are being violated add the cost for CVQE
		if newClusterForS_1 ~= newClusterForS_2
			GMLV( newClusterForS_1 ).item = [ GMLV( newClusterForS_1 ).item; newClusterForS_2 ]; 
			GMLV( newClusterForS_1 ).cost = [ GMLV( newClusterForS_1 ).cost; distanceBetweenCentroids(newClusterForS_1, newClusterForS_2) ]; 
		end
	end


	%cannot link treatment
	for c=CLs
		s_1 = constraints(c, 1);
		s_2 = constraints(c, 2);
		c_j = idx(s_1);
		c_n = idx(s_2);
		
		%constraint already being satisfied
		if c_j ~= c_n
			continue;
		end

		%need to calculate all the possible assignments
		%and choose the one that causes the smallest CVQE

		newClusterForS_1 = 0;		
		newClusterForS_2 = 0;		
		smallerCost = inf;
		for k1=1:k
			for k2=k1:k

				if k1 == k2
					%for the cases of violation of constraints calculates the cost involved
					minDist = min(distanceBetweenCentroids(k2,:));
					costInitial = 0.5*minDist;
				else
					costInitial = 0;
				end

				%check the cost of putting s_1 in k1 and s_2 in k2
				cost = costInitial + 0.5*distances(s_1,k1) + 0.5*distances(s_2,k2);
				if(cost < smallerCost)
					newClusterForS_1 = k1;
					newClusterForS_2 = k2;
					smallerCost = cost;
				end

				%check the cost of putting s_1 in k2 and s_2 in k1
				cost = costInitial + 0.5*distances(s_1,k2) + 0.5*distances(s_2,k1);

				if(cost < smallerCost)
					newClusterForS_1 = k2;
					newClusterForS_2 = k1;
					smallerCost = cost;
				end
			end
		end
		idx(s_1) = newClusterForS_1;
		idx(s_2) = newClusterForS_2;

		%if constraints are being violated add the cost for CVQE
		if newClusterForS_1 == newClusterForS_2
			[minDist h_gl] = min(distanceBetweenCentroids(newClusterForS_1,:));
			GCLV( newClusterForS_1 ).item = [ GCLV( newClusterForS_1 ).item; h_gl ]; 
			GCLV( newClusterForS_1 ).cost = [ GCLV( newClusterForS_1 ).cost; minDist ]; 
		end

	end

	for c=1:k
		members = find(idx == c);
		coordsMembers = sum( X(members,:) );
		coordsGMLV = sum( centroids(GMLV(c).item,:) );
		coordsGCLV = sum( centroids(GCLV(c).item,:) );
		z_j = length(members) + length(GMLV(c).item) + length(GCLV(c).item);
		centroids(c,:) = (coordsMembers + coordsGMLV + coordsGCLV) ./ z_j;
	end

	CVQE = zeros(k,1);
	for c=1:k
		members = find(idx == c);
		if(size(members,1)==0)
			return
		end
		CVQE(c) = 0.5*sum(distances(members, c));
		sumML = 0;
		sumCL = 0;
		for costForViolation=GMLV(c).cost'
			sumML = sumML + costForViolation;
		end
		for costForViolation=GCLV(c).cost'
			sumCL = sumCL + costForViolation;
		end
		CVQE(c) = CVQE(c) + 0.5*sumML + 0.5*sumCL;
	end
	CVQE = sum(CVQE);
	
	iter = iter + 1;
end

time=toc(tstart);
