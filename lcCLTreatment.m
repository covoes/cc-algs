function [centroids] = lcCLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo)
%lcCLTreatment Takes care of CL constraints when processing X(i,:)
% X : data
% i : object being processed
% otherObject : the object that has a CL constraint with X(i,:)
% j : closest centroid to X(i,:)
% r : closest centroid to X(otherObject,:)
% centrois : current centroids positiosn
% learnRate: rate for attraction between the prototype and a vector
% delearnRate: rate for a prototype moving away from a vector
% distancesXi : distance between centroids and X(i,:)
% distancesXo : distance between centroids and X(otherObject,:)


if r == j
	if distancesXo(j) > distancesXi(j)
		farthest = otherObject;
		tmp = distancesXo;
	else
		farthest = i;
		tmp = distancesXi;
	end
	tmp(j) = NaN;
	%find the nearest neighbor of the farthest object
	[nearDist v] = min(tmp);
	
	costVio = 0.5*(distancesXi(j)+distancesXo(j)+nearDist);
	if farthest == i
		costPutInNeighbor = 0.5*(distancesXo(j)+distancesXi(v));
	else
		costPutInNeighbor = 0.5*(distancesXi(j)+distancesXo(v));
	end

	if costVio < costPutInNeighbor
		centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
		%move neighbor prototype towards farthest object
		centroids(v,:) = centroids(v,:) + learnRate * (X(farthest,:) - centroids(v,:));
	else
		centroids(v,:) = centroids(v,:) + learnRate * (X(farthest,:) - centroids(v,:));
		if farthest == otherObject 
			centroids(j,:) = centroids(j,:) - delearnRate * (X(otherObject,:) - centroids(j,:));
			centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
		end
	end

else
	centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
end
