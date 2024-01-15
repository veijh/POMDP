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

//#pragma GCC optimize(3)

#include "iostream"
#include <iomanip>
#include "vector"
#include "omp.h"
#include "Eigen/Eigen"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unordered_map"
#include "POMDP.h"
#include "single_UAV_maze.h"
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
    FILE* node_file = fopen("../node_gdb.csv", "r");
    FILE* edge_file = fopen("../edge_gdb.csv", "r");

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
    vector<vector<int>> unk_part{{1,3,2,4}};
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

    vector<int> src = {0}, dst = {5}, key_node;
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
//            if(end == begin) continue;
            double dis = map.dijkstra(begin, {end}, path);
            if(path.size() != 0){
                if(begin == 1 && end == 2) continue;
                if(begin == 2 && end == 1) continue;
                if(begin == 3 && end == 4) continue;
                if(begin == 4 && end == 3) continue;
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

    for(auto item:state_map){
        cout << item << ", ";
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
            if(real_node == 5){
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
    MATSL::read_binary("../output.bin.19", alpha_vector);
    cout << alpha_vector << endl;

    Eigen::VectorXf _belief_state(state_dim);
//    Eigen::VectorXf node_belief_state = pow(0.5, doors_num)*Eigen::VectorXf::Ones(all_condition_num);
    Eigen::VectorXf node_belief_state = Eigen::VectorXf::Zero(all_condition_num);
    node_belief_state(1) = 1;
    cout << node_belief_state << endl;

    cout << compact_map.get_node_num() << endl;

    for(int node = 0; node < compact_map.get_node_num(); node++){
        _belief_state.setConstant(0);
        _belief_state.middleRows(all_condition_num*node, all_condition_num) = node_belief_state;

        Eigen::VectorXf adv_belief_state(1 + state_dim);
        adv_belief_state.block(1, 0, state_dim, 1) = _belief_state;
//        cout << adv_belief_state;
//        cout << "start to calculate result" << endl;
        Eigen::VectorXf result = alpha_vector * adv_belief_state;
//        cout << "start to search max_v" << endl;
        double max_v = result.maxCoeff();
//        cout << "start to match the best action" << endl;
        vector<int> best_actions;
        for(int i = 0; i < alpha_vector.rows(); i++){
            if(abs(max_v - result(i)) < 0.01) {
                if (find(best_actions.begin(), best_actions.end(),(int)alpha_vector(i,0)) == best_actions.end()) {
                    best_actions.push_back((int)alpha_vector(i,0));
                }
            }
        }

        cout << state_map[node] << ": ";
        for(auto act:best_actions){
            if(act < fc_compact_map.adj_table[state_map[node]].edge_list.size()){
                auto header = fc_compact_map.adj_table[state_map[node]].edge_list.begin();
                for(int mv = 0; mv < act; mv++, header++);
                cout << header->first << "; ";
            }
            else
            {
                cout << "act: " << act << ", ?? ";
            }
        }
        cout << endl;
    }
}
