import io
import igraph
import sys


def a2a_update(file1, file2,output_file):
# def a2a_update(file1, file2):
    f = open(file1, "r",encoding='utf-8')
    lines = f.readlines()
    f.close()
  

    a1 = {}
    for line in lines:
        templist = line.strip().split(" ")
        # print templist
        if len(templist) < 3:
            continue
        first_field = templist[0]
        third_field = templist[-1]
        module_name = ' '.join(templist[1:-1])
        # print(module_name)
        if not module_name in a1:
            a1[module_name] = []
        a1[module_name].append(third_field)

    f = open(file2, "r",encoding='utf-8')
    lines = f.readlines()
    f.close()

    a2 = {}
    for line in lines:
        templist = line.strip().split(" ")
        # print templist
        if len(templist) < 3:
            continue
        first_field = templist[0]
        third_field = templist[-1]
        module_name = ' '.join(templist[1:-1])
        # print(module_name)
        if not module_name in a2:
            a2[module_name] = []
        a2[module_name].append(third_field)

    # 获取 a1 和 a2 所有文件的集合
    files_a1 = set(f for files in a1.values() for f in files)
    files_a2 = set(f for files in a2.values() for f in files)

    # 只在 a1 中存在的文件
    delefile = files_a1 - files_a2
    print(f"delefile文件：{delefile}")
    # 只在 a2 中存在的文件
    addfile = files_a2 - files_a1
    print(f"addfile文件：{addfile}")
    match_type = []
    for i in range(0, len(a1)):
        match_type.append(0)
        pass

    for i in range(0, len(a2)):
        match_type.append(1)
        pass

    # print len(match_type)

    edge_dict = {}
    # 记录 a1 和 a2 中所有模块的交集文件
    a1_a2_intersection = set()
    i = 0
    for temp1 in range(0, len(a1)):
        for temp2 in range(0, len(a2)):
            edge_dict[(temp1, temp2 + len(a1))] = len(set(tuple(a1.values())[temp1]) & set(tuple(a2.values())[temp2]))
            files_a1 = set(tuple(a1.values())[temp1])
            files_a2 = set(tuple(a2.values())[temp2])
            intersection = files_a1 & files_a2
            if intersection:
                a1_a2_intersection.update(intersection)
            # result = set(tuple(a1.values())[temp1]) & set(tuple(a2.values())[temp2])

    g = igraph.Graph()
    g.add_vertices(len(a1) + len(a2))
    g.add_edges(edge_dict.keys())
    # 新增  为了绘图加的
    types = [0] * len(a1) + [1] * len(a2)
    g.vs["type"] = types
    count = 0
    # 初始化 match_type，使其长度与 graph 中的节点数量相匹配
    match_type = [0] * len(g.vs)

    # 遍历 edge_dict.keys()，为 match_type 中的元素赋值
    for temp in edge_dict.keys():
        if temp[0] < len(a1):
            match_type[temp[0]] = 1
        else:
            match_type[temp[0] + len(a1) - len(g.vs)] = 1

    # 确保所有 match_type 元素都被赋值
    if len(match_type) != len(g.vs):
        raise ValueError("The length of match_type does not match the number of vertices in the graph")

    edge_dict_real = tuple(edge_dict.values())
    matching = g.maximum_bipartite_matching(match_type, edge_dict_real)
    # a1 到 a2 的最大匹配对
    a1_to_a2 = []
    for i in range(len(a1)):
        j = matching.match_of(i)
        if j is not None and j >= len(a1):
            # i 是 a1 的索引，j-len(a1) 是 a2 的索引
            a1_to_a2.append((list(a1.keys())[i], list(a2.keys())[j - len(a1)]))
            print(f"a1的{i}与a2的{j - len(a1)}匹配")
    # print("a1_to_a2的值")
    # 获取匹配的边
    matched_edges = matching.edges()
    # 设置颜色
    edge_colors = ["gray"] * g.ecount()
    for edge in matched_edges:
        edge_colors[edge.index] = "red"

    vertex_colors = ["blue" if g.vs[i]["type"] == 0 else "green" for i in range(g.vcount())]
    layout = g.layout_bipartite()

    # 绘制并保存图片
    # plot = igraph.plot(g, layout=layout, edge_color=edge_colors,vertex_label=range(g.vcount()), bbox=(400, 400), margin=20)
    # plot.save("bipartite_matching_result.png")

    removeA = []
    moveAB = []
    addB = []

    for i in range(0, len(a1) + len(a2)):
        if i < len(a1):
            if matching.match_of(i) == None:
                removeA.append(i)
            else:
                moveAB.append((i, matching.match_of(i)))
        else:
            if matching.match_of(i) == None:
                addB.append(i)

    # 软件项目的两个版本中簇映射关系
    print("removeA的值")
    print(removeA)
    print("moveAB的值")
    print(moveAB)
    print("addB的值")
    print(addB)
    res = {}

    mto = 0
    dis_sum = abs(len(a1) - len(a2))+ 2*abs(sum(len(v) for v in a1.values()) - sum(len(v) for v in a2.values()))
    a2a_sum = 0
    for i in removeA:
        # print("前一个版本中 模块" + str(i) + " removeA中的距离是" + str(1 + 2 * len(tuple(a1.values())[i])))
        res[(list(a1.keys())[i], -1)] = 1 + 2 * len(tuple(a1.values())[i])
        # 中间是修改的
        mto += 2 * len(tuple(a1.values())[i])
        # 中间是修改的

        # 原本代码中的
        # mto+=len(tuple(a1.values())[i])
        # 原本代码中的

        # 加1是因为该模块在后一个版本中不存在 在addC中存在一个操作
        mto += 1
        pass
    for i in addB:
        # print("后一个版本中 模块" + str(i - len(a1)) + " addB中的距离是" + str(
        #     1 + 2 * len(tuple(a2.values())[i - len(a1)])))
        # print((1 + len(tuple(a2.values())[i-len(a1)])) / 5034)
        # 中间是修改的
        mto += 2 * len(tuple(a2.values())[i - len(a1)])
        # 中间是修改的
        res[(-1, list(a2.keys())[i - len(a1)])] = 1 + 2 * len(tuple(a2.values())[i - len(a1)])
        # 原本代码中的
        # mto+=len(tuple(a2.values())[i-len(a1)])
        # 原本代
        # 原本代码中的
        # mto+=len(tuple(a2.values())[i-len(a1)])
        # 原本代码中的
        # 加1是因为该模块在前一个版本中不存在 在addC中存在一个操作
        mto += 1
    for temp in moveAB:
        res[(list(a1.keys())[temp[0]], list(a2.keys())[temp[1] - len(a1)])] = len(set(tuple(a1.values())[temp[0]])-delefile) - 2*edge_dict[temp] + len(set(tuple(a2.values())[temp[1] - len(a1)])-addfile)
        mto += len(set(tuple(a1.values())[temp[0]])-delefile) - 2*edge_dict[temp] + len(set(tuple(a2.values())[temp[1] - len(a1)])-addfile)
        dis_sum += len(set(tuple(a1.values())[temp[0]])-delefile) - 2*edge_dict[temp] + len(set(tuple(a2.values())[temp[1] - len(a1)])-addfile)
    # mto(a0,a1) = |C_a1| + 2 * |E_a1|
    # 中间是修改的
    aco1 = 2 * sum(map(len, a1.values())) + len(a1)
    aco2 = 2 * sum(map(len, a2.values())) + len(a2)
    # 中间是修改的

    # 原本代码中的
    # aco1=sum(map(len,a1.values()))+len(a1)
    # aco2=sum(map(len,a2.values()))+len(a2)
    # 原本代码中的
    # print(aco1)
    # print(aco2)
    # print("mto的值: " + str(mto))
    # print("dis_sum的值: " + str(dis_sum))
    a2a = 1 - float(dis_sum) / (float(aco1) + float(aco2))
    # print("a2a的值: " + str(a2a))
    count = 0
    for key,value in res.items():
        if mto == 0.0:
            res[key] = "0.00000"
        else:
            res[key]= f"{value/mto:.5f}"
        count += float( res[key])
        print(f"res[key]: {key} 的值是 {res[key]}")


    with open(output_file, 'w', encoding='utf-8') as file:
        for key, value in res.items():
            key1, key2 = key
            print(f"a2a_backup Value:{key1},{key2},{value}\n")
            file.write(f"{key1},{key2},{value}\n")
    print(a2a)
    return res, a2a, mto

if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print("Usage: python git_diff_file_modify_aquire.py <rsf1_file> <rsf2_file> ")
    #     sys.exit(1)
    # rsf1_rsf = sys.argv[1]
    # rsf2_rsf = sys.argv[2]
    # output_file = sys.argv[3]
    # a2a_update(rsf1_rsf,rsf2_rsf,output_file)

    rsf1_rsf = r"../sema_results/libuv-1.49.0/libuv-1.49.0_NamedClusters.rsf"
    rsf2_rsf = r"../sema_results/libuv-1.49.1/libuv-1.49.1_NamedClusters.rsf"
    # with open(rsf1_rsf, "r", encoding="utf-8") as f:
    #     rsf1_rsf = f.readlines()
    # with open(rsf2_rsf, "r", encoding="utf-8") as f:
    #     rsf2_rsf = f.readlines()
    a2a_update(rsf1_rsf,rsf2_rsf,"../sema_results/libuv_a2a_update.json")



