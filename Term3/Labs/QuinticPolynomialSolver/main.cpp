#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "Eigen/Dense"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

vector<double> JMT(vector< double> start, vector <double> end, double T)
{
    double si = start[0];
    double si_dot = start[1];
    double si_double_dot = start[2];
    double sf = end[0];
    double sf_dot = end[1];
    double sf_double_dot = end[2];

    double result_1 = sf - (si + si_dot * T + (si_double_dot * T * T ) / 2);
    double result_2 = sf_dot - (si_dot + si_double_dot * T);
    double result_3 = sf_double_dot - si_double_dot;

    MatrixXd s_matrix = MatrixXd(1, 3);
    s_matrix << result_1, result_2, result_3;
    MatrixXd T_matrix = MatrixXd(3, 3);
    T_matrix << pow(T, 3), pow(T, 4), pow(T, 5),
                3 * pow(T, 2), 4 * pow(T, 3), 5 * pow(T, 4),
                6 * pow(T, 1), 12 * pow(T, 2), 20 * pow(T, 3);

    MatrixXd T_inverse = T_matrix.inverse();
    MatrixXd s_transpose = s_matrix.transpose();
    MatrixXd output = T_inverse * s_transpose;

    double a_0 = si;
    double a_1 = si_dot;
    double a_2 = si_double_dot / 2;
    double a_3 = output(0);
    double a_4 = output(1);
    double a_5 = output(2);

    return {a_0, a_1, a_2, a_3, a_4, a_5};
}

bool close_enough(vector< double > poly, vector<double> target_poly, double eps=0.01) {


    if(poly.size() != target_poly.size())
    {
        cout << "your solution didn't have the correct number of terms" << endl;
        return false;
    }
    for(int i = 0; i < poly.size(); i++)
    {
        double diff = poly[i]-target_poly[i];
        if(abs(diff) > eps)
        {
            cout << "at least one of your terms differed from target by more than " << eps << endl;
            return false;
        }

    }
    return true;
}
    
struct test_case {
    
        vector<double> start;
        vector<double> end;
        double T;
};

vector< vector<double> > answers = {{0.0, 10.0, 0.0, 0.0, 0.0, 0.0},{0.0,10.0,0.0,0.0,-0.625,0.3125},{5.0,10.0,1.0,-3.0,0.64,-0.0432}};

int main() {

    //create test cases

    vector< test_case > tc;

    test_case tc1;
    tc1.start = {0,10,0};
    tc1.end = {10,10,0};
    tc1.T = 1;
    tc.push_back(tc1);

    test_case tc2;
    tc2.start = {0,10,0};
    tc2.end = {20,15,20};
    tc2.T = 2;
    tc.push_back(tc2);

    test_case tc3;
    tc3.start = {5,10,2};
    tc3.end = {-30,-20,-4};
    tc3.T = 5;
    tc.push_back(tc3);

    bool total_correct = true;
    for(int i = 0; i < tc.size(); i++)
    {
        vector< double > jmt = JMT(tc[i].start, tc[i].end, tc[i].T);
        bool correct = close_enough(jmt,answers[i]);
        total_correct &= correct;
    }
    if(!total_correct)
    {
        cout << "Try again!" << endl;
    }
    else
    {
        cout << "Nice work!" << endl;
    }

    return 0;
}