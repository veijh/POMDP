//
// Created by wjh on 23-12-29.
//

#ifndef CPP_TEST_POMDP_H
#define CPP_TEST_POMDP_H

#include "iostream"
#include <iomanip>
#include "vector"
#include "algorithm"
#include "unordered_map"
#include "map"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "set"
#include "cstdlib"
#include <sys/time.h>
#include <zlib.h>
#include "fstream"

#define EPS (1e-5)

using namespace std;

class POMDP{
private:
    // dimension of state_dim space: S
    int state_dim;
    // dimension of action space: A
    int act_dim;
    // dimension of observation space: O
    int obs_dim;

    /* using ptr instead of var can avoid unnecessary copy */
    // state_dim transition probabilities taking specific action: AxSxS
//    vector<Eigen::MatrixXf> trans_vec;
    vector<Eigen::SparseMatrix<float>> trans_vec;
    // immediate rewards: SxA
    Eigen::MatrixXf rwd_s_a;
    // observation probabilities: OxS
//    Eigen::MatrixXf p_obs_in_s;
    Eigen::SparseMatrix<float> p_obs_in_s;
    // belief points: (1+S)xN
//    Eigen::MatrixXf belief_points;
    Eigen::SparseMatrix<float> belief_points;
    // each point corresponds with an alpha vector: Nx(1+S)
    Eigen::MatrixXf alpha_vector;
public:
    // init POMDP
    POMDP(const vector<Eigen::MatrixXf> &transition, const Eigen::MatrixXf &r_s_a, const Eigen::MatrixXf &p_o_s);
    // use PBVI to solve POMDP, _belief_points: SxN
    void PBVI(Eigen::SparseMatrix<float> _belief_points, int horizon_len);
    // select best actions according to belief state, _belief_points: SxN
    vector<int> select_action(Eigen::VectorXf _belief_state);
    // use bayesian inference to update the belief state, belief: Sx1
    Eigen::VectorXf bayesian_filter(Eigen::VectorXf _belief_state, int _obs);
};

namespace MATSL{
    /*
    template<class Derived>
    void write_binary(const std::string &filename,
                      const Eigen::PlainObjectBase<Derived> &matrix)
    {
        typedef typename Derived::Index Index;
        typedef typename Derived::Scalar Scalar;

        gzFile out = gzopen(filename.c_str(), "wb");
        Index rows=matrix.rows(), cols=matrix.cols();

        gzwrite(out, (char*) (&rows), sizeof(Index));
        gzwrite(out, (char*) (&cols), sizeof(Index));
        gzwrite(out, (char*) matrix.data(), rows*cols*sizeof(Scalar) );
        gzclose(out);
    }

    template<class Derived>
    void read_binary(const std::string &filename,
                     Eigen::PlainObjectBase<Derived> &matrix)
    {
        typedef typename Derived::Index Index;
        typedef typename Derived::Scalar Scalar;

        gzFile in = gzopen(filename.c_str(), "rb");
        Index rows=0, cols=0;
        gzread(in, (char*) (&rows),sizeof(Index));
        gzread(in, (char*) (&cols),sizeof(Index));
        matrix.resize(rows, cols);
        gzread(in, (char*) matrix.data(), rows*cols*sizeof(Scalar) );
        gzclose(in);
    }
    */

    template<class Matrix>
    void write_binary(const char* filename, const Matrix& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
        out.write((char*) (&rows), sizeof(typename Matrix::Index));
        out.write((char*) (&cols), sizeof(typename Matrix::Index));
        out.write((char*) matrix.data(), rows*cols*sizeof(typename Matrix::Scalar) );
        out.close();
    }
    template<class Matrix>
    void read_binary(const char* filename, Matrix& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        typename Matrix::Index rows=0, cols=0;
        in.read((char*) (&rows),sizeof(typename Matrix::Index));
        in.read((char*) (&cols),sizeof(typename Matrix::Index));
        matrix.resize(rows, cols);
        in.read( (char *) matrix.data() , rows*cols*sizeof(typename Matrix::Scalar) );
        in.close();
    }

} // MATSL::

#endif //CPP_TEST_POMDP_H
