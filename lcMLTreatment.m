function [centroids] = lcMLTreatment(X, i, otherObject, j, r, centroids, learnRate, delearnRate, distancesXi, distancesXo)
%lcMLTreatment Takes care of ML constraints when processing X(i,:)
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


if r == j
	centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
else
	costVio = 0.5*(distancesXi(j)+distancesXo(r))+0.25*(distancesXi(r)+distancesXo(j));
	costPutInJ = 0.5*(distancesXi(j)+distancesXo(j));
	costPutInR = 0.5*(distancesXi(r)+distancesXo(r));
	[vals idx] = min([ costVio costPutInJ costPutInR ]);
	if idx == 1
		%better to violate
		centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
		%move centroids towards the objects they do not have
		centroids(j,:) = centroids(j,:) + learnRate * (X(o,:) - centroids(j,:));
		centroids(r,:) = centroids(r,:) + learnRate * (X(i,:) - centroids(r,:));
	else if idx == 2
		%better to put both in j
		centroids(j,:) = centroids(j,:) + learnRate * (X(i,:) - centroids(j,:));
		centroids(j,:) = centroids(j,:) + learnRate * (X(otherObject,:) - centroids(j,:));
		centroids(r,:) = centroids(r,:) - delearnRate * (X(otherObject,:) - centroids(r,:));
	else
		%better to put both in r
		centroids(r,:) = centroids(r,:) + learnRate * (X(i,:) - centroids(r,:));
	end

end

end
