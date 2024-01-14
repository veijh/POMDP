//
// Created by 魏小爷 on 2023/9/18.
//
#include "maze_map.h"

int Map::get_node_num() {
    node_num = adj_table.size();
    return node_num;
}

void Map::add_edge(const int &id1, const int &id2) {
    if (adj_table[id1].edge_list.find(id2) == adj_table[id1].edge_list.end()) {
        adj_table[id1].edge_list[id2] = (adj_table[id1].pos - adj_table[id2].pos).norm();
    } else {
        cout << "duplicate edge: (" << id1 << ", " << id2 << ")" << endl;
    }
    if (adj_table[id2].edge_list.find(id1) == adj_table[id2].edge_list.end()) {
        adj_table[id2].edge_list[id1] = (adj_table[id1].pos - adj_table[id2].pos).norm();
    } else {
        cout << "duplicate edge: (" << id1 << ", " << id2 << ")" << endl;
    }
    return;
}

void Map::add_edge(const int &id1, const int &id2, const double &dis){
    if (adj_table[id1].edge_list.find(id2) == adj_table[id1].edge_list.end()) {
        adj_table[id1].edge_list[id2] = dis;
    }
    if (adj_table[id2].edge_list.find(id1) == adj_table[id2].edge_list.end()) {
        adj_table[id2].edge_list[id1] = dis;
    }
    return;
}

void Map::del_edge(const int &id1, const int &id2) {
    adj_table[id1].edge_list.erase(id2);
    adj_table[id2].edge_list.erase(id1);
}

void Map::dijkstra(const int &src_id, const vector<int> &dst_id) {
    get_node_num();
    // 未确定的最短路径长度
    vector<double> distance2node(node_num, INF_D);
    // 确定的最短路径长度
    vector<double> min_distance2node(node_num, INF_D);
    // 确定的最短路径长度
    vector<bool> path_find(node_num, false);
    // 上一节点
    vector<int> path2node(node_num, -1);

    // 确定距离当前顶点最近的最短路径
    distance2node.at(src_id) = 0;
    min_distance2node.at(src_id) = 0;
    double min_dis = *min_element(distance2node.begin(), distance2node.end());
    while (min_dis > INF_D + 0.1 || min_dis < INF_D - 0.1) {
        min_dis = *min_element(distance2node.begin(), distance2node.end());
        // 确定最短路径及长度
        int min_id = min_element(distance2node.begin(), distance2node.end()) - distance2node.begin();
        path_find.at(min_id) = true;
        min_distance2node.at(min_id) = distance2node.at(min_id);

        // 拓展min_id节点的edge
        for (auto &edge: adj_table[min_id].edge_list) {
            // 跳过确定的最短路径点
            if (path_find.at(edge.first)) {
                continue;
            }
            // 如果经过min_id的距离更小，更新该节点的上一节点
            if (edge.second + min_distance2node.at(min_id) < distance2node.at(edge.first)) {
                path2node.at(edge.first) = min_id;
                distance2node.at(edge.first) = edge.second + min_distance2node.at(min_id);
            }
        }
        // 不再考虑该节点
        distance2node.at(min_id) = INF_D;
    }

    for (auto &dst: dst_id) // 路径表
    {
        int count = 0;
        cout << src_id << " to " << dst << ":";
        int id = dst;
        // 这里可以补充判断id == src_id
        while (count < node_num && id != src_id) {
            id = path2node.at(id);
            if (id == -1) {
                cout << "no path";
                break;
            } else {
                cout << "<-" << id;
                count++;
            }
        }
        cout << endl;
        cout << "distance = " << min_distance2node.at(dst) << "m" << endl;
    }
}

double Map::dijkstra(const int &src_id, const vector<int> &dst_id, vector<int> &path) {
    get_node_num();
    if (!path.empty()) {
        path.clear();
    }
    // 未确定的最短路径长度
    vector<double> distance2node(node_num, INF_D);
    // 确定的最短路径长度
    vector<double> min_distance2node(node_num, INF_D);
    // 确定的最短路径长度
    vector<bool> path_find(node_num, false);
    // 上一节点
    vector<int> path2node(node_num, -1);

    // 确定距离当前顶点最近的最短路径
    distance2node.at(src_id) = 0;
    min_distance2node.at(src_id) = 0;
    double min_dis = *min_element(distance2node.begin(), distance2node.end());
    while (min_dis > INF_D + 0.1 || min_dis < INF_D - 0.1) {
        min_dis = *min_element(distance2node.begin(), distance2node.end());
        // 确定最短路径及长度
        int min_id = min_element(distance2node.begin(), distance2node.end()) - distance2node.begin();
        path_find.at(min_id) = true;
        min_distance2node.at(min_id) = distance2node.at(min_id);

        // 拓展min_id节点的edge
        for (auto &edge: adj_table[min_id].edge_list) {
            // 跳过确定的最短路径点
            if (path_find.at(edge.first)) {
                continue;
            }
            // 如果经过min_id的距离更小，更新该节点的上一节点
            if (edge.second + min_distance2node.at(min_id) < distance2node.at(edge.first)) {
                path2node.at(edge.first) = min_id;
                distance2node.at(edge.first) = edge.second + min_distance2node.at(min_id);
            }
        }
        // 不再考虑该节点
        distance2node.at(min_id) = INF_D;
    }

    // 最近的目标点
    int min_dst_id = dst_id.at(0);
    double min_dst_dis = min_distance2node.at(0);
    for (auto id: dst_id) {
        if (min_distance2node.at(id) < min_dst_dis) {
            min_dst_id = id;
            min_dst_dis = min_distance2node.at(id);
        }
    }
    cout << src_id << " to " << min_dst_id << ":";
    int id = min_dst_id;
    if(min_dst_id == src_id){
        path.push_back(src_id);
        cout << " stay" << endl;
        return 0.0;
    }
    id = path2node.at(id);
    if (id == -1) {
        cout << "no path" << endl;
        return -1.0;
    } else {
        path.push_back(min_dst_id);
        path.push_back(id);
        cout << "<-" << id;
        while (id != src_id) {
            id = path2node.at(id);
            path.push_back(id);
            cout << "<-" << id;
        }
        reverse(path.begin(), path.end());
    }
    cout << "; distance = " << min_distance2node.at(min_dst_id) << "m" << endl;
    return min_distance2node.at(min_dst_id);
}

