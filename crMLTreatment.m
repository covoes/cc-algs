function [centroids w] = crMLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo, w, gamma)
%CRMLTreatment Takes care of ML constraints when processing X(i,:)
% X : data
% i : object being processed
% otherObject : the object that has a ML constraint with X(i,:)
% j : closest centroid to X(i,:)
% r : closest centroid to X(otherObject,:)
% centrois : current centroids positiosn
% learnRate: rate for attraction between the prototype and a vector
% delearnRate: rate for a prototype moving away from a vector
% distancesXi : distance between centroids and X(i,:)
% distancesXo : distance between centroids and X(otherObject,:)

middle = 0.5*(X(i,:) + X(otherObject,:));
centroids(r,:) = centroids(r,:) + learnRate * (middle - centroids(r,:));
w(r) = w(r) + 1;

if r ~= j
	centroids(j,:) = centroids(j,:) - delearnRate * (X(i,:) - centroids(j,:));
else
	distancesXi(j) = NaN;
	[nearDist n] = min(distancesXi);
	centroids(n,:) = centroids(n,:) - delearnRate * (X(i,:) - centroids(n,:));
end


end
