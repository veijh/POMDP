//
// Created by wjh on 23-12-18.
//
#include "POMDP.h"

POMDP::POMDP(const vector<Eigen::MatrixXf> &transition, const Eigen::MatrixXf &r_s_a, const Eigen::MatrixXf &p_o_s) {
    act_dim = transition.size();
    state_dim = r_s_a.rows();
    obs_dim = p_o_s.rows();
    cout << "[INFO] state_dim space dim: " << state_dim << ", "
         << "action space dim: " << act_dim << ", "
         << "obs_dim space dim: " << obs_dim << endl;

    struct timeval t1{},t2{};
    double timeuse;
    cout << "[LOG] convert T(a,s,s) to sparse mat.";
    gettimeofday(&t1,nullptr);
    for(int i = 0; i < transition.size(); i++) {
        trans_vec.push_back(transition[i].sparseView());
    }
    gettimeofday(&t2,nullptr);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout<<" using time = " << timeuse << " s" << endl;  //(in sec)

    cout << "[LOG] copy r(s,a).";
    gettimeofday(&t1,nullptr);
    rwd_s_a = r_s_a;
    gettimeofday(&t2,nullptr);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout<<" using time = " << timeuse << " s" << endl;  //(in sec)

    cout << "[LOG] convert p(o,s) to sparse mat.";
    gettimeofday(&t1,nullptr);
    p_obs_in_s = p_o_s.sparseView();
    gettimeofday(&t2,nullptr);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout<<" using time = " << timeuse << " s" << endl;  //(in sec)
}

void POMDP::PBVI(Eigen::SparseMatrix<float> _belief_points, int horizon_len) {
//    Eigen::MatrixXf augmented_belief = Eigen::MatrixXf::Zero(1 + state_dim, _belief_points.cols());
//    augmented_belief.bottomRows(state_dim) = _belief_points;
    cout << "[LOG] start to solve POMDP using PBVI" << endl;
//    struct timeval t1{},t2{};
//    double timeuse;
//    cout << "convert belief_point to sparse mat.";
//    gettimeofday(&t1,nullptr);
//    belief_points = augmented_belief.sparseView();
    belief_points = _belief_points;
//    cout << Eigen::MatrixXf(belief_points) << endl;
//    gettimeofday(&t2,nullptr);
//    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
//    cout<<" using time = " << timeuse << " s" << endl;  //(in sec)

    int points_num = belief_points.cols();
    cout << "[INFO] the num of belief points: " << points_num << endl;

    alpha_vector.conservativeResize(points_num, 1 + state_dim);
    alpha_vector.setConstant(0);

    cout << "[LOG] initialize tmp" << endl;
    // In fact, the type of tmp can be "vector<vector<Eigen::MatrixXf>>". AxOxNx(1+S)
    vector<vector<Eigen::MatrixXf>> tmp;
    tmp.resize(act_dim);
    for (int action = 0; action < act_dim; ++action) {
        tmp[action].resize(obs_dim);
        for (int z = 0; z < obs_dim; ++z) {
            tmp[action][z].conservativeResize(points_num, 1 + state_dim);
        }
    }

    // calculate the value function
    for (int horizon = 0; horizon < horizon_len; horizon++) {
        cout << "iteration: " << horizon << endl;
        Eigen::MatrixXf new_alpha;
        if(horizon % 10 == 9){
            cout << "output alpha vector" << endl;
            MATSL::write_binary("../output.bin." + to_string(horizon), alpha_vector);
        }

        cout << "calculate tmp.";
        struct timeval t1{},t2{};
        double timeuse;
        gettimeofday(&t1,nullptr);
        // 这一段可以并行计算
        // tmp一共有points_num * action * observation 个元素
        // belief数量
        #pragma omp parallel for num_threads(24)
        for (int k = 0; k < points_num; k++) {
            // 动作
            for (int action = 0; action < act_dim; action++) {
                // 观测
                for (int z = 0; z < obs_dim; ++z) {
                    // 第一列为 action
#ifdef DBG
                    struct timeval t1{},t2{};
                    double timeuse;
                    gettimeofday(&t1,nullptr);
#endif
                    tmp[action][z](k,0) = 0;
                    tmp[action][z].row(k).rightCols(state_dim) = alpha_vector.row(k).rightCols(state_dim).cwiseProduct(p_obs_in_s.row(z))
                                                               * trans_vec[action].transpose();
#ifdef DBG
                    gettimeofday(&t2,nullptr);
                    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
                    cout<<"time = " << timeuse << endl;  //(in sec)
#endif
                }
            }
        }
        gettimeofday(&t2,nullptr);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout<<" using time = " << timeuse << " s" << endl;  //(in sec)

        cout << "update alpha." << endl;
        gettimeofday(&t1,nullptr);
        // Vbar(b)是可以求解的，因此每个belief点对应action个可能的alpha_vector
        new_alpha.conservativeResize(act_dim * points_num, 1 + state_dim);
        new_alpha.setConstant(0);

        #pragma omp parallel for num_threads(24)
        // 计算 new alpha, 该段计算时间远大于tmp的计算，当前瓶颈在cpu核心数，考虑迁移到gpu下cuda计算
        // belief点
        for(int k = 0; k < points_num; k++){
            // 对于某个指定动作
            for (int action = 0; action < act_dim; action++) {
                // 对于某个指定观测
                for(int z = 0; z < obs_dim; z++){
                    // 计算V(b|z)
                    // 查找使得alpha*b最大的alpha. prod: Nx(1+S)x(1+S)x1 = Nx1
                    Eigen::VectorXf prod = tmp[action][z] * belief_points.col(k);
                    int index = 0;
                    prod.maxCoeff(&index);

                    // 求和得到Vbar
                    new_alpha.row(action + act_dim * k) += tmp[action][z].row(index);
                }
                new_alpha(action + act_dim * k, 0) = action;

                // reward可以写成R(s,a)矩阵
                // 状态
                new_alpha.block(action + act_dim*k, 1, 1, state_dim) += rwd_s_a.col(action).transpose();
            }
            // 从new_alpha的action中选择最优动作，更新alpha_vector
            int best_action = 0;
            Eigen::VectorXf result = new_alpha.block(act_dim * k, 0, act_dim, 1 + state_dim) * belief_points.col(k);
            result.maxCoeff(&best_action);
            alpha_vector.row(k) = new_alpha.row(best_action + act_dim * k);
        }

        gettimeofday(&t2,nullptr);
        timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
        cout<<" using time = " << timeuse << " s" << endl;  //(in sec)
//        cout << "alpha_vector:" << endl << alpha_vector << endl;
    }
//    cout << "alpha_vector:" << endl << alpha_vector << endl;
}

vector<int> POMDP::select_action(const Eigen::VectorXf& _belief_state) {
    Eigen::VectorXf adv_belief_state = Eigen::VectorXf::Zero(1 + state_dim);
    adv_belief_state.block(1, 0, state_dim, 1) = _belief_state;
    Eigen::SparseMatrix<float> sparse = adv_belief_state.sparseView();
    Eigen::VectorXf result = alpha_vector * sparse;
    float max_v = result.maxCoeff();
    vector<int> best_actions;
    for(int i = 0; i < alpha_vector.rows(); i++){
        if(abs(max_v - result(i)) < EPS) {
            if (find(best_actions.begin(), best_actions.end(),(int)alpha_vector(i,0)) == best_actions.end()) {
                best_actions.push_back((int)alpha_vector(i,0));
            }
        }
    }
    return best_actions;
}

Eigen::VectorXf POMDP::bayesian_filter(const Eigen::VectorXf& _belief_state, int _obs) {
    Eigen::VectorXf new_belief = p_obs_in_s.row(_obs).transpose().cwiseProduct(_belief_state);// (p_obs_in_s.row(_obs).transpose().array() * adv_belief_state.array()).matrix();
    if(new_belief.sum() < EPS){
        cout << "[ERROR] no possible state" << endl;
        return new_belief;
    }
    new_belief /= new_belief.sum();
    return new_belief;
}

void POMDP::import_alpha(const Eigen::MatrixXf &alpha_mat) {
    alpha_vector = alpha_mat;
}
