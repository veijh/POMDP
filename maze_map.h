//
// Created by 魏小爷 on 2023/9/18.
//

#ifndef MAZE_MAP_H
#define MAZE_MAP_H

const double INF_D = 999.0;

#include <iostream>
#include <vector>
#include <list>
#include <algorithm>
#include "unordered_map"
#include "Eigen/Eigen"
#include "Eigen/Dense"

using namespace std;

typedef struct Node {
    Eigen::Vector2d pos;
    unordered_map<int, double> edge_list;
}Node;

class Map
{
private:
    // 该拓扑地图的总节点数量
    int node_num;
public:
    // 容器中存放所有节点信息
    unordered_map<int, Node> adj_table;
    // 获取该地图节点总数
    int get_node_num();
    // 添加边
    void add_edge(const int &id1, const int &id2);
    void add_edge(const int &id1, const int &id2, const double &dis);
    // 删去边
    void del_edge(const int &id1, const int &id2);
    // dijkstra寻路，终端输出结果
    void dijkstra(const int &src_id, const vector<int> &dst_id);
    // dijkstra寻路，结果存放到path中
    double dijkstra(const int &src_id, const vector<int> &dst_id, vector<int> &path);
};

#endif
