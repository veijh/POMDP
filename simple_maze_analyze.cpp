//
// Created by wjh on 24-1-2.
//

#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL
#endif

#ifndef EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_2
#endif

#define CRASH_REWARD (-999)

#pragma GCC optimize(3)

#include "iostream"
#include <iomanip>
#include "vector"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unordered_map"
#include "POMDP.h"
#include "maze_map.h"
#include "string"

using namespace std;

// return x^y
int my_pow(const int& x, const int& y){
    int ans = 1;
    for(int i = 0; i < y; i++){
        ans *= x;
    }
    return ans;
}

// dim: C(n,r) x r
vector<vector<int>> generateCombinations(int n, int r) {
    vector<vector<int>> output;
    if(r == 0) return output;
    vector<int> combination(r, 0);

    // 初始化组合的初始状态
    for (int i = 0; i < r; ++i) {
        combination[i] = i;
    }

    while (combination[0] <= n - r) {
        // 输出当前组合
//        for (int num : combination) {
//            cout << num << " ";
//        }
//        cout << endl;
        output.push_back(combination);

        // 生成下一个组合
        int i = r - 1;
        while (i >= 0 && combination[i] == n - r + i) {
            --i;
        }

        if (i >= 0) {
            ++combination[i];
            for (int j = i + 1; j < r; ++j) {
                combination[j] = combination[j - 1] + 1;
            }
        } else {
            break;  // 已生成所有组合
        }
    }
    return output;
}

// get the bit of value, index:[....76543210]
unsigned int get_bit(int value, int r){
    return (value >> r) & 1;
}

int main() {
#if _OPENMP
    cout << "support openmp " << endl;
#else
    cout << "not support openmp" << endl;
#endif
    omp_set_num_threads(6);

    // read topo from file
    FILE* node_file = fopen("../node.csv", "r");
    FILE* edge_file = fopen("../edge.csv", "r");

    if(node_file == nullptr) {
        cout << "[ERROR] fail to open node file!" << endl;
        return 0;
    }

    if(edge_file == nullptr) {
        cout << "[ERROR] fail to open edge file!" << endl;
        return 0;
    }

    Map map;
    double x = 0.0, y = 0.0;
    int id = 0;
    while (~fscanf(node_file, "%lf,%lf,%d", &x, &y, &id)) {
        map.adj_table[id].pos << x,y;
    }

    int id1 = 0, id2 = 0;
    while (~fscanf(edge_file, "%d,%d", &id1, &id2)) {
        map.add_edge(id1, id2);
    }
    cout << "[INFO] the number of nodes is " << map.get_node_num() << endl;

    fclose(node_file);
    fclose(edge_file);

    /*
    for(auto node:map.adj_table){
        cout << "node " << node.first << " >> ";
        for(auto item:node.second.edge_list){
            cout << item.first << ": " << item.second << "; ";
        }
        cout << endl;
    }
    */

    // unknown door
    vector<vector<int>> unk_part{{6,7,8},{9,10,11},{13,12,14},{33,35,34},{23,22,24},
                                 {16,17,15,18},{25,26,27,28}};
    const int doors_num = unk_part.size();
    const int all_condition_num = my_pow(2, doors_num);
    cout << "[INFO] the number of unknown doors is " << doors_num << endl;

    unordered_map<int, int> var;
    vector<int> unk_node;
    for(int i = 0; i < doors_num; i++){
        for(auto item:unk_part[i]){
            var[item] = i;
            unk_node.push_back(item);
        }
    }

    vector<int> src = {0, 1, 2}, dst = {3, 4, 5}, key_node;
    key_node.insert(key_node.end(), src.begin(), src.end());
    key_node.insert(key_node.end(), dst.begin(), dst.end());
    key_node.insert(key_node.end(), unk_node.begin(), unk_node.end());

    Map compact_map;
    for(int node:key_node){
        compact_map.adj_table[node].pos = map.adj_table[node].pos;
    }
    cout << "[INFO] the number of key nodes is " << compact_map.get_node_num() << endl;

    vector<int> path;
    for(int begin:key_node){
        for (int end:key_node){
            if(end == begin) continue;
            double dis = map.dijkstra(begin, {end}, path);
            if(path.size() != 0){
                compact_map.add_edge(begin, end, dis);
            }
        }
    }

    /*
    for(auto node:compact_map.adj_table){
        cout << "node " << node.first << " >> ";
        for(auto item:node.second.edge_list){
            cout << item.first << ": " << item.second << "; ";
        }
        cout << endl;
    }
     */

    cout << "[LOG] determine the act dim" << endl;
    Map fc_compact_map(compact_map);
    for(auto item:unk_part){
        int size = item.size();
        fc_compact_map.add_edge(item[0], item[1]);
        fc_compact_map.add_edge(item[size-2], item[size-1]);
    }

    for(auto node:fc_compact_map.adj_table){
        cout << "node " << node.first << ": ";
        int count = 0;
        for(auto item:node.second.edge_list){
            cout << "act[" << count << "]:" << item.first << "; ";
            count++;
        }
        cout << endl;
    }

    int act_dim = 0;
    for(auto item:fc_compact_map.adj_table){
        if(item.second.edge_list.size() > act_dim) {
            act_dim = item.second.edge_list.size();
        }
    }
    cout << "[INFO] act_dim is " << act_dim << endl;

    // all kinds of possible structure
    vector<Map> all_compact_map(all_condition_num, compact_map);

    for(int i = 0; i < all_compact_map.size(); i++){
        for(int bit = 0; bit < doors_num; bit++){
            int id_c1, id_c2;
            if( ((i >> bit) & 1) == 0){
                id_c1 = unk_part[bit][0];
                id_c2 = unk_part[bit][1];
            }
            else{
                id_c1 = unk_part[bit][unk_part[bit].size()-2];
                id_c2 = unk_part[bit][unk_part[bit].size()-1];
            }

            double dis = (compact_map.adj_table[id_c1].pos - compact_map.adj_table[id_c2].pos).norm();
            all_compact_map[i].adj_table[id_c1].edge_list[id_c2] = dis;
            all_compact_map[i].adj_table[id_c2].edge_list[id_c1] = dis;
        }
    }

    int obs_dim = 3, state_dim = compact_map.get_node_num()*all_condition_num+1;
    cout << "[INFO] obs_dim = " << obs_dim << ", state_dim = " << state_dim  << endl;
    vector<int> state_map(compact_map.get_node_num());
    unordered_map<int,int> inv_state_map;
    for(int i = 0; i < state_map.size(); i++){
        auto header = compact_map.adj_table.begin();
        for(int j = 0; j < i; j++, header++);
        state_map[i] = header->first;
        inv_state_map[header->first] = i;
    }

    for(int i = 0; i < state_map.size(); i++){
        cout << "s" << i << ": " << state_map[i];
        if(i % 10 == 9){
            cout << endl;
        }
        else{
           cout << " || ";
        }
    }
    cout << endl;

//    for(auto item:inv_state_map){
//        cout << item.first << ", " << item.second << endl;
//    }

    // init POMDP
    vector<Eigen::MatrixXf> tran_vec;
    Eigen::MatrixXf tran(state_dim, state_dim);
    tran.setConstant(0);
    for(int i = 0; i < act_dim; i++){
        tran_vec.push_back(tran);
    }
    Eigen::MatrixXf reward = Eigen::MatrixXf::Zero(state_dim, act_dim);
    Eigen::MatrixXf p_o_s = Eigen::MatrixXf::Zero(obs_dim, state_dim);
    for(int s = 0; s < state_dim; s++){
        for(int act = 0; act < act_dim; act++){
            // the last state is a crashed state
            if(s == state_dim - 1) {
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // every node else has all_condition_num states
            int real_node = state_map[s/all_condition_num];
            // dst node is an absorbed state
            if(real_node == 3 || real_node == 4 || real_node == 5){
                tran_vec[act](s, s) = 1;
                reward(s, act) = 0;
                continue;
            }
            // normal node
            int dst_id = 0;
            if(act < fc_compact_map.adj_table[real_node].edge_list.size()){
                auto header = fc_compact_map.adj_table[real_node].edge_list.begin();
                for(int mv = 0; mv < act; mv++, header++);
                dst_id = header->first;
                // search for dst_id according to s%all_condition_num map
                auto it = all_compact_map[s%all_condition_num].adj_table[real_node].edge_list;
                if(it.find(dst_id) != it.end()){
                    tran_vec[act](s, all_condition_num*inv_state_map[dst_id] + s%all_condition_num) = 1;
                    reward(s, act) = -header->second;
                }
                else
                {
                    tran_vec[act](s, state_dim-1) = 1;
                    reward(s, act) = CRASH_REWARD;
                }
            }
            else{
                tran_vec[act](s, state_dim-1) = 1;
                reward(s, act) = CRASH_REWARD;
            }
        }
        // p_o_s
        // s is unk_node
        if (s != state_dim-1 && var.find(state_map[s/all_condition_num]) != var.end()) {
            if ( ( ( (s % all_condition_num) >> var[state_map[s/all_condition_num]]) & 1) == 0) {
                p_o_s(0, s) = 1;
                p_o_s(1, s) = 0;
                p_o_s(2, s) = 0;
            }
            else {
                p_o_s(0, s) = 0;
                p_o_s(1, s) = 1;
                p_o_s(2, s) = 0;
            }
        }
        // normal node, i.e. src, dst
        else {
            p_o_s(0, s) = 0;
            p_o_s(1, s) = 0;
            p_o_s(2, s) = 1;
        }
    }

//    ofstream file("p_o_s.csv");
//    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
//    file << p_o_s.format(CSVFormat);
//    ofstream file("r_s_a.csv");
//    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
//    file << reward.format(CSVFormat);

//    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
//    for(int i = 0; i < tran_vec.size(); i++){
//        ofstream file("tran_" + to_string(i) + ".csv");
//        file << tran_vec[i].format(CSVFormat);
//    }

    Eigen::MatrixXf alpha_vector;
    MATSL::read_binary("../7door/output.bin.59", alpha_vector);

    POMDP PBVI(tran_vec, reward, p_o_s);
    PBVI.import_alpha(alpha_vector);

//    Eigen::VectorXf _belief_state(state_dim);
//    Eigen::VectorXf node_belief_state = pow(0.5, doors_num)*Eigen::VectorXf::Ones(all_condition_num);
//
//    for(int node = 0; node < compact_map.get_node_num(); node++){
//        _belief_state.setConstant(0);
//        _belief_state.middleRows(all_condition_num*node, all_condition_num) = node_belief_state;
//
//        Eigen::VectorXf adv_belief_state(1 + state_dim);
//        adv_belief_state.setConstant(0);
//        adv_belief_state.block(1, 0, state_dim, 1) = _belief_state;
//        Eigen::VectorXf result = alpha_vector * adv_belief_state;
//        float max_v = result.maxCoeff();
//        vector<int> best_actions;
//        for(int i = 0; i < alpha_vector.rows(); i++){
//            if(abs(max_v - result(i)) < 0.001) {
//                if (find(best_actions.begin(), best_actions.end(),(int)alpha_vector(i,0)) == best_actions.end()) {
//                    best_actions.push_back((int)alpha_vector(i,0));
//                }
//            }
//        }
//
//        cout << state_map[node] << ": ";
//        for(auto act:best_actions){
//            if(act < fc_compact_map.adj_table[state_map[node]].edge_list.size()){
//                auto header = fc_compact_map.adj_table[state_map[node]].edge_list.begin();
//                for(int mv = 0; mv < act; mv++, header++);
//                cout << "act: " << act << " = " << header->first << "; ";
//            }
//            else
//            {
//                cout << "act: " << act << ", ?? ";
//            }
//        }
//        cout << endl;
//    }

    // start from different nodes
    for(int i = 0; i < src.size(); i++){
        // test distance under all conditions
        int init_node = src[i];
        for(int j = 0; j < all_compact_map.size(); j++){
            // initial belief state
            Eigen::VectorXf belief = Eigen::VectorXf::Zero(state_dim);
            belief.middleRows(all_condition_num * inv_state_map[init_node], all_condition_num) = pow(0.5, doors_num) * Eigen::VectorXf::Ones(all_condition_num);
            // total_dis
            double total_dis = 0.0;
            int cur_node = init_node;
            // while not arriving dst
            while(std::find(dst.begin(), dst.end(), cur_node) == dst.end()){
                // receive observation
                int z = -1;
                int true_state = all_condition_num * inv_state_map[cur_node] + j;
//                cout << "receive observation..." << endl;
                p_o_s.col(true_state).maxCoeff(&z);
//                cout << "obs is " << z << endl;
                // update belief
//                cout << "update belief..." << endl;
                belief = PBVI.bayesian_filter(belief, z);
                Eigen::VectorXf belief_block = belief.middleRows(all_condition_num * inv_state_map[cur_node], all_condition_num);
                // select action
//                cout << "select action..." << endl;
                vector<int> actions = PBVI.select_action(belief);
                if(actions.empty()){
                    cout << "[ERROR] no valid action" << endl;
                    return 1;
                }
                int next_best_node = -1;
                if(actions.size() > 1){
                    cout << "[WARNING] action num more than once: ";
                    float next_min_dis = 999.0;
                    for(int& item:actions){
                        if(item < fc_compact_map.adj_table[cur_node].edge_list.size()){
                            auto header = fc_compact_map.adj_table[cur_node].edge_list.begin();
                            for(int mv = 0; mv < item; mv++, header++);
                            cout << "to node " << header->first << "; ";
                            if(reward(true_state, item) < next_min_dis){
                                next_min_dis = reward(true_state, item);
                                next_best_node = header->first;
                            }
                        }
                    }
                    cout << endl;
                }
                int best_action = next_best_node;
//                cout << "best action is " << best_action << endl;
                // state transition
//                cout << "state transition..." << endl;
                int next_state = -1;
                tran_vec[best_action].row(true_state).maxCoeff(&next_state);
//                if(best_action < fc_compact_map.adj_table[cur_node].edge_list.size()){
//                    auto header = fc_compact_map.adj_table[cur_node].edge_list.begin();
//                    for(int mv = 0; mv < best_action; mv++, header++);
//                    cur_node = header->first;
//                }
                // check
                if(j != next_state % all_condition_num){
                    cout << "[ERROR] wrong state transition" << endl;
                    return 1;
                }
                cur_node = state_map[next_state/all_condition_num];
                cout << ">> node " << cur_node << " ";
                // update belief state
                belief.setZero();
                belief.middleRows(all_condition_num * inv_state_map[cur_node], all_condition_num) = belief_block;
                // add distance
                total_dis += reward(true_state, best_action);
                cout << "total dis = " << total_dis << endl;
            }
            cout << "[RESULT] map " << j << ":" << "from " << init_node << " to " << cur_node << ", total dis = " << total_dis << endl;
        }
    }
}
