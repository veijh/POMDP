//
// Created by wjh on 23-12-18.
//
#include "POMDP.h"

POMDP::POMDP(vector<Eigen::MatrixXd> transition, Eigen::Matrix2Xd r_s_a, Eigen::Matrix2Xd p_o_s) {
    act_dim = transition.size();
    state_dim = r_s_a.rows();
    obs_dim = p_o_s.rows();
    cout << "state_dim space dim: " << state_dim << ", "
         << "action space dim: " << act_dim << ", "
         << "obs_dim space dim: " << obs_dim << endl;
    trans_vec = transition;
    rwd_s_a = r_s_a;
    p_obs_in_s = p_o_s;
}

void POMDP::PBVI(Eigen::MatrixXd belief_points, int horizon_len) {
    int points_num = belief_points.cols();

    alpha_vector.conservativeResize(points_num, 1 + state_dim);
    alpha_vector.setConstant(0);

    // calculate the value function
    for (int horizon = 0; horizon < horizon_len; horizon++) {
        cout << "iteration: " << horizon << endl;
        Eigen::MatrixXd new_alpha;
        vector<vector<vector<Eigen::Matrix<double, 1, 1+state_dim>>>> tmp;
        tmp.resize(points_num);
        for (int row = 0; row < points_num; row++) {
            tmp[row].resize(4);
            for (int action = 0; action < 4; action++) {
                tmp[row][action].resize(2);
            }
        }

        // 这一段可以并行计算
        // tmp一共有points_num * action * observation 个元素
        // belief数量
        for (int k = 0; k < points_num; k++) {
            // 动作
            for (int action = 0; action < 4; action++) {
                // 观测
                for (int z = 0; z < 2; ++z) {
                    // 第一列为 action
                    tmp[k][action][z](0,0) = 0;
                    tmp[k][action][z].rightCols(state_dim) = (alpha_vector.row(k).rightCols(state_dim).array() * p_obs_in_s.row(z).array()).matrix()
                                                               * trans_vec[action].transpose();
                }
            }
        }

        // Vbar(b)是可以求解的，因此每个belief点对应action个可能的alpha_vector
        new_alpha.conservativeResize(4 * points_num, 1 + state_dim);
        new_alpha.setConstant(0);

        // belief点
        for(int k = 0; k < points_num; k++){
            // 对于某个指定动作
            for (int action = 0; action < 4; action++) {
                // 对于某个指定观测
                for(int z = 0; z < 2; z++){
                    // 计算V(b|z)
                    // 查找使得alpha*b最大的alpha
                    vector<double> prod_vec;
                    for(int new_k = 0; new_k < points_num; new_k++){
                        double prod = tmp[new_k][action][z] * belief_points.col(k);
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
                    if (s == 0 || s == 1 || s == 2 || s == 3) reward = 0;
                    new_alpha(action + 4*k, 1+s) += reward;
                }
            }

            // 从action中选择最优动作，更新alpha_vector
            int best_action = 0;
            auto result = new_alpha.block(4*k, 0, 4, 1+state_dim) * belief_points.col(k);
            result.maxCoeff(&best_action);
            alpha_vector.row(k) = new_alpha.row(best_action + 4*k);
        }
    }
    cout << alpha_vector << endl;

}
