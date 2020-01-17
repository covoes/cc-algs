function [centroids, w] = crCLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo, w, gamma)
%CRCLTreatment Takes care of CL constraints when processing X(i,:)
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


distancesXi(j) = NaN;
[nearDist n] = min(distancesXi);

if r ~= j
	%RPCL
	centroids(n,:) = centroids(n,:) - delearnRate * (X(i,:) - centroids(n,:));
	centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
	w(j) = w(j) + 1;
else
	%attract neighbor
	centroids(n,:) = centroids(n,:) + learnRate * (X(i,:) - centroids(n,:));
	centroids(j,:) = centroids(j,:) - delearnRate * (X(i,:) - centroids(j,:));
	w(n) = w(n) + 1;

end


end
