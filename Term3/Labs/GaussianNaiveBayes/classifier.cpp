#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

using namespace std;

/**
 * Initializes GNB
 */
GNB::GNB() {

}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/
	//cout << "Theta: " << _theta.size() << endl;
	if (data.size() != labels.size())
	{
	    cout << "Data size is different from label size" << endl;
	    return;
	}
	// make class buckets:
	int bucketCount = 3;
	_classCount.clear();
	_theta.clear();
	_sigma.clear();
	vector<vector<double>> sum;
	for (auto _ = 0; _ < bucketCount; ++_)
	{
	    _classCount.push_back(0);
	    _classPrior.push_back(0);
	    vector<double> t;
	    vector<double> s;
	    vector<double> v;
	    for (auto k = 0; k < data[0].size(); ++k){
	        t.push_back(0.0);
	        s.push_back(0.0);
	        v.push_back(0.0);
	    }
    	_theta.push_back(t);
    	_sigma.push_back(s);
    	sum.push_back(v);
	}
	for(auto i = 0; i < data.size(); ++i)
	{
	    int classIndex = 0;
	    switch(labels[i][0])
	    {
	        case 'l':
	            classIndex = 0;
	            break;
	        case 'k':
	            classIndex = 1;
	            break;
	        case 'r':
	            classIndex = 2;
	            break;
	        default:
	            cout << "Unknown label value: " << labels[i] << endl;
	            return;
	    }
	    _classCount[classIndex]++;
	    for (auto j = 0; j < data[i].size(); ++j)
	    {
	        sum[classIndex][j] += data[i][j];
	    }
	}
	for (auto bucket = 0; bucket < bucketCount; ++bucket)
	{
	    auto n = _classCount[bucket];
	    for (auto k = 0; k < _theta[bucket].size(); ++k)
	    {
	        _theta[bucket][k] = sum[bucket][k] / n;
	    }
	}
	for(auto i = 0; i < data.size(); ++i)
	{
	    int classIndex = 0;
	    switch(labels[i][0])
	    {
	        case 'l':
	            classIndex = 0;
	            break;
	        case 'k':
	            classIndex = 1;
	            break;
	        case 'r':
	            classIndex = 2;
	            break;
	        default:
	            cout << "Unknown label value: " << labels[i] << endl;
	            return;
	    }
	    auto n = _classCount[classIndex];
	    for (auto j = 0; j < data[i].size(); ++j)
	    {
	        double x = data[i][j];
	        double mu = _theta[classIndex][j];
	        _sigma[classIndex][j] += pow((x - mu), 2);
	    }
	}
	for (auto bucket = 0; bucket < bucketCount; ++bucket)
	{
	    //cout << "Bucket = " << bucket << endl;
	    auto n = _classCount[bucket];
	    for (auto k = 0; k < _sigma[bucket].size(); ++k)
	    {
	        //cout << "Dimension = " << k << endl;
	        _sigma[bucket][k] = _sigma[bucket][k] / n;
	        //cout << "Sigma = " << _sigma[bucket][k] << endl;
	        //cout << "Theta = " << _theta[bucket][k] << endl;
	    }
	}
}

string GNB::predict(vector<double> v)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
	*/
	vector<double> p = {0, 0, 0};
	int bucketCount = 3;
	for (auto bucket = 0; bucket < bucketCount; ++bucket)
	{
	    double f = 0;
	    for (auto d = 0; d < v.size(); ++d)
	    {
    	    double muC = _theta[bucket][d];
    	    double sigmaC = _sigma[bucket][d];
    	    double twoSigmaC = 2 * sigmaC;
    	    double rootOfPiTwoSigmaC = sqrt(M_PI * twoSigmaC);
	        f += (1/rootOfPiTwoSigmaC) * exp(-pow((v[d] - muC), 2) / twoSigmaC);
	    }
	    p[bucket] = f;
	}
	int maxIndex = 0;
	for (auto bucket = 0; bucket < bucketCount; ++bucket)
	{
	    if (p[bucket] > p[maxIndex])
	    {
	        maxIndex = bucket;
	    }
	}

	return this->possible_labels[maxIndex];

}
