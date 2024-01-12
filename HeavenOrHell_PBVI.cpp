//
// Created by wjh on 23-12-18.
//
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#pragma GCC optimize(3)

#include "iostream"
#include <iomanip>
#include "vector"
#include "algorithm"
#include "unordered_map"
#include "map"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "set"
#include "cstdlib"

using namespace std;

enum ACTION {
    UP = 0, DOWN, LEFT, RIGHT
};

class INDEX_MAP {
private:
    int row, col, world, obs;
public:
    INDEX_MAP(int _row, int _col, int _world, int _obs) : row(_row), col(_col), world(_world), obs(_obs) {};

    int get_index(int i, int j, int _world) {
        return _world + world * j + world * col * i;
    }
};

int main() {
//    Eigen::setNbThreads(10)
    omp_set_num_threads(6);
    double instant_reward = -1;
    INDEX_MAP state_index(5, 4, 2, 2);

    const int total_state = 16+1;

    // 观测的似然
    Eigen::MatrixXd p_obs_state(3, total_state);
    p_obs_state.setConstant(0);
    p_obs_state.bottomRows(1) = Eigen::RowVectorXd ::Ones(total_state);
    p_obs_state(0, 14) = 1;
    p_obs_state(1, 14) = 0;
    p_obs_state(2, 14) = 0;
    p_obs_state(0, 15) = 0;
    p_obs_state(1, 15) = 1;
    p_obs_state(2, 15) = 0;

    // 状态转移
    vector<Eigen::Matrix<double, total_state, total_state>> trans_vec;
    Eigen::Matrix<double, total_state, total_state> transition(total_state, total_state);
    transition.setConstant(0);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    trans_vec.push_back(transition);
    for (int i = 0; i < total_state; i++) {
        switch (i / 2) {
            case 0:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 1:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
            case 2:
                trans_vec[UP](i, i - 4) = 1;
                trans_vec[DOWN](i, 16) = 1;
                trans_vec[LEFT](i, 16) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 3:
                trans_vec[UP](i, 16) = 1;
                trans_vec[DOWN](i, i + 4) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 4:
                trans_vec[UP](i, i - 6) = 1;
                trans_vec[DOWN](i, 16) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, 16) = 1;
                break;
            case 5:
                trans_vec[UP](i, i - 4) = 1;
                trans_vec[DOWN](i, i + 2) = 1;
                trans_vec[LEFT](i, 16) = 1;
                trans_vec[RIGHT](i, 16) = 1;
                break;
            case 6:
                trans_vec[UP](i, i - 2) = 1;
                trans_vec[DOWN](i, 16) = 1;
                trans_vec[LEFT](i, 16) = 1;
                trans_vec[RIGHT](i, i + 2) = 1;
                break;
            case 7:
                trans_vec[UP](i, 16) = 1;
                trans_vec[DOWN](i, 16) = 1;
                trans_vec[LEFT](i, i - 2) = 1;
                trans_vec[RIGHT](i, 16) = 1;
                break;
            case 8:
                trans_vec[UP](i, i) = 1;
                trans_vec[DOWN](i, i) = 1;
                trans_vec[LEFT](i, i) = 1;
                trans_vec[RIGHT](i, i) = 1;
                break;
        }
    }

    // PBVI的核心
    const int point_num = 3*(total_state-1)/2;
    Eigen::MatrixXd possible_state(2,3);
    possible_state << 1, 0, 0.5,
            0, 1, 0.5;
    int active_num = 0;
    // 信念点的集合 N+1 x point_num
    Eigen::MatrixXd belief_point(1+total_state, point_num);
    belief_point.setConstant(0);
    for(int i = 0; i < (total_state-1)/2; i++){
        belief_point.block(1+2*i,3*i,2,3) = possible_state;
    }
    // 每一个信念点对应一个alpha_vector(行向量) point_num x N+1
    // 第一列是动作
    Eigen::MatrixXd alpha_vector(point_num, 1+total_state);
    alpha_vector.setConstant(0);

    // 计算Vh
    for (int horizon = 0; horizon < 10; horizon++) {
        cout << "iteration: " << horizon << endl;
        Eigen::MatrixXd new_alpha;
        vector<vector<vector<Eigen::Matrix<double, 1, 1+total_state>>>> tmp;
        tmp.resize(point_num);
        for (int row = 0; row < point_num; row++) {
            tmp[row].resize(4);
            for (int action = 0; action < 4; action++) {
                tmp[row][action].resize(3);
            }
        }

        // 这一段可以并行计算
        // tmp一共有point_num * action * observation 个元素
        // belief数量
        for (int k = 0; k < point_num; k++) {
            // 动作
            for (int action = 0; action < 4; action++) {
                // 观测
                for (int z = 0; z < 3; ++z) {
                    // 第一列为 action
                    tmp[k][action][z](0,0) = 0;
                    tmp[k][action][z].rightCols(total_state) = (alpha_vector.row(k).rightCols(total_state).array() * p_obs_state.row(z).array()).matrix()
                                        * trans_vec[action].transpose();
                }
            }
        }

        // Vbar(b)是可以求解的，因此每个belief点对应action个可能的alpha_vector
        new_alpha.conservativeResize(4 * point_num, 1 + total_state);
        new_alpha.setConstant(0);

        // belief点
        for(int k = 0; k < point_num; k++){
            // 对于某个指定动作
            for (int action = 0; action < 4; action++) {
                // 对于某个指定观测
                for(int z = 0; z < 3; z++){
                    // 计算V(b|z)
                    // 查找使得alpha*b最大的alpha
                    vector<double> prod_vec;
                    for(int new_k = 0; new_k < point_num; new_k++){
                        double prod = tmp[new_k][action][z] * belief_point.col(k);
                        prod_vec.push_back(prod);
                    }
                    int index = max_element(prod_vec.begin(), prod_vec.end()) - prod_vec.begin();

                    // 求和得到Vbar
                    new_alpha.row(action + 4 * k) += tmp[index][action][z];
                }
                new_alpha(action + 4 * k, 0) = action;

                // reward可以写成R(s,a)矩阵
                // 状态
                for (int s = 0; s < total_state; ++s) {
                    double reward = -1;
                    if (s == 4 && action == UP) reward = 100;
                    if (s == 5 && action == UP) reward = -100;
                    if (s == 8 && action == UP) reward = -100;
                    if (s == 9 && action == UP) reward = 100;
                    if (trans_vec[action](s, 16) == 1) reward = -200;
                    if (s == 0 || s == 1 || s == 2 || s == 3 || s == 16) reward = 0;
                    new_alpha(action + 4*k, 1+s) += reward;
                }
            }

            // 从action中选择最优动作，更新alpha_vector
            int best_action = 0;
            auto result = new_alpha.block(4*k, 0, 4, 1+total_state) * belief_point.col(k);
            result.maxCoeff(&best_action);
            alpha_vector.row(k) = new_alpha.row(best_action + 4*k);
        }
    }
    cout << alpha_vector << endl;

    vector<set<int>> policy;
    for(int s = 0; s < total_state/2; s++){
        Eigen::VectorXd belief(1+total_state);
        belief.setConstant(0);
        belief(2*s + 1) = 0.5;
        belief(2*s+1 + 1) = 0.5;
        Eigen::Index index;
        auto result = alpha_vector*belief;
        double max_v = result.maxCoeff();
        set<int> tmp;
        for(int i = 0; i < alpha_vector.rows(); i++){
            if(max_v == result(i)){
                if (tmp.find(alpha_vector(i,0)) == tmp.end()) {
                    tmp.insert(alpha_vector(i,0));
                }
            }
        }
        policy.push_back(tmp);
    }

    int index = 0;
    for(auto i:policy){
        switch (index) {
            case 1:
            case 5:
            case 6:
                cout << setw(5) << " ";
                break;
            default:
                break;
        }
        string output;
        for(auto j:i){
            switch (j) {
                case UP:
                    output += "^";
                    break;
                case LEFT:
                    output += "<";
                    break;
                case RIGHT:
                    output += ">";
                    break;
                case DOWN:
                    output += "v";
                    break;
            }
        }
        cout << setw(5) << output;
        switch (index) {
            case 1:
            case 4:
            case 5:
            case 7:
                cout << endl;
                break;
            default:
                break;
        }
        index++;
    }

//    cout << "hello";
}
