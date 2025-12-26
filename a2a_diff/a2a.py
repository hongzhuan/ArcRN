import io
import igraph
import sys


def a2a(file1, file2):
	f = open(file1, "r")
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
			pass
		a1[module_name].append(third_field)
		print("a1 is:")
		print(a1)
	# for i in a1.keys():
	# 	print("i的种类有："+i)
	# for temp1,temp2 in a1.items():
	# 	print temp1
	# 	print temp2

	# print len(a1)

	f = open(file2, "r")
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
			pass
		a2[module_name].append(third_field)
	# print("a2 is:")
	# print(a2)
	# for temp1,temp2 in a2.items():
	# 	print (temp1)
	# 	print (temp2)
	# print("a2.values() is:")
	# print(a2.values())
	# print len(a2)

	match_type = []
	for i in range(0, len(a1)):
		match_type.append(0)
		pass

	for i in range(0, len(a2)):
		match_type.append(1)
		pass

	# print len(match_type)

	edge_dict = {}
	i = 0
	for temp1 in range(0, len(a1)):
		for temp2 in range(0, len(a2)):
			edge_dict[(temp1, temp2 + len(a1))] = len(set(tuple(a1.values())[temp1]) & set(tuple(a2.values())[temp2]))
			# print("a2簇的文件:"+str(set(tuple(a2.values())[temp2])))
			# print(set(tuple(a1.values())[temp1]) & set(tuple(a2.values())[temp2]))
			# print("edge_dict值"+str([temp1,temp2]) + str(edge_dict[(temp1,temp2+len(a1))]))
			# print(f"edge_dict值{[temp1,temp2]} is {edge_dict[(temp1,temp2+len(a1))]}")
			pass
	# print("a1的第"+str(i)+"簇中的文件:" + str(set(tuple(a1.values())[temp1])))
	# print(len(set(tuple(a1.values())[temp1])))
	# duplicates=set(x for x in tuple(a1.values())[temp1] if tuple(a1.values())[temp1].count(x)>1)
	# print("a1的第"+str(i)+"簇中重复的文件:"+str(duplicates))
	# i = i + 1
	# for temp1 in range(0, len(a1)):
	# 	for temp2 in range(0, len(a2)):
	# 		for value1 in a1.values():
	# 			for value2 in a2.values():
	# 				if value1 == value2:
	# 					if (temp1, temp2 + len(a1)) not in edge_dict:
	# 						edge_dict[(temp1, temp2 + len(a1))] = 1
	# 					else:
	# 						edge_dict[(temp1, temp2 + len(a1))] += 1

	g = igraph.Graph()
	g.add_vertices(len(a1) + len(a2))

	g.add_edges(edge_dict.keys())

	# 新增  为了绘图加的
	types = [0] * len(a1) + [1] * len(a2)
	g.vs["type"] = types

	count = 0

	# 对中间结果的 key: value 进行打印  两个版本前后不同簇之间对应的关系
	for key_edge in edge_dict.keys():
		print("key: " + str(key_edge), end=" ")
		print("value: " + str(edge_dict[key_edge]), end=" ")
		count += 1
		if count % 8 == 0:
			print()

	# count=0
	# for values_edge in edge_dict.values():
	# 	print(values_edge,end= " ")
	# 	if count%10==0:
	# 		print()
	# for temp in edge_dict.keys():
	# 	g.add_edges(temp)

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
	# print("match_type:")
	# print(match_type)
	# print("edge_weight:")
	# print(edge_dict.values())
	edge_dict_real = tuple(edge_dict.values())
	print("edge_dict_real:" + str(edge_dict_real))
	matching = g.maximum_bipartite_matching(match_type, edge_dict_real)
	# print("matching的类型:"+ str(type(matching)))
	print(g.vs["type"])

	# 获取匹配的边
	matched_edges = matching.edges()
	# 输出匹配的边到控制台
	for edge in matched_edges:
		print(f"Matched edge: {edge.source} - {edge.target}")
	#
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
				pass
			else:
				moveAB.append((i, matching.match_of(i)))
				pass
			pass
		else:
			if matching.match_of(i) == None:
				addB.append(i)
			pass
		pass

	# 软件项目的两个版本中簇映射关系
	print("removeA的值")
	print(removeA)
	print("moveAB的值")
	print(moveAB)
	print("addB的值")
	print(addB)
	res = {}

	mto = 0
	dis_sum = 0
	for i in removeA:
		print("前一个版本中 模块" + str(i) + " removeA中的距离是" + str(1 + 2 * len(tuple(a1.values())[i])))
		res[(list(a1.keys())[i], -1)] = 1 + 2 * len(tuple(a1.values())[i])
		# print((1+len(tuple(a1.values())[i]))/5034)
		# 中间是修改的
		# mto += 2 * len(tuple(a1.values())[i])
		# 中间是修改的

		# 原本代码中的
		mto+=len(tuple(a1.values())[i])
		# 原本代码中的

		# 加1是因为该模块在后一个版本中不存在 在addC中存在一个操作
		mto += 1
		dis_sum += mto
		pass
	for i in addB:
		print("后一个版本中 模块" + str(i - len(a1)) + " addB中的距离是" + str(
			1 + 2 * len(tuple(a2.values())[i - len(a1)])))
		# print((1 + len(tuple(a2.values())[i-len(a1)])) / 5034)
		# 中间是修改的
		# mto += 2 * len(tuple(a2.values())[i - len(a1)])
		mto += len(tuple(a2.values())[i - len(a1)])
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
		dis_sum += mto
		pass
	for temp in moveAB:
		print("(" + str(temp[0]) + "," + str(temp[1] - len(a1)) + ") " + "在moveAB中的距离是" + str(
			len(tuple(a1.values())[temp[0]]) - edge_dict[temp] + len(tuple(a2.values())[temp[1] - len(a1)]) - edge_dict[
				temp]))
		res[(list(a1.keys())[temp[0]], list(a2.keys())[temp[1] - len(a1)])] = len(tuple(a1.values())[temp[0]]) - edge_dict[temp] + len(tuple(a2.values())[temp[1] - len(a1)]) - edge_dict[temp]
		# print((len(tuple(a1.values())[temp[0]])-edge_dict[temp]+len(tuple(a2.values())[temp[1]-len(a1)])-edge_dict[temp])/5034)
		mto += len(tuple(a1.values())[temp[0]]) - edge_dict[temp] + len(tuple(a2.values())[temp[1] - len(a1)]) - edge_dict[temp]
		dis_sum += mto
		# print(len(tuple(a1.values())[temp[0]]))
		# print(len(tuple(a2.values())[temp[1]-len(a1)]))
		# print(tuple(a1.values())[temp[0]])
		# print(tuple(a2.values())[temp[1]-len(a1)])
		# print(edge_dict[temp])
		pass

	# print("累计mto的值")
	# print(mto)

	# sorted() 返回一个按值排序的元组列表
	sorted_items = sorted(res.items(), key=lambda item: item[1], reverse=True)
	for key, value in sorted_items:
		print("res中" + f'{key}: {value}')

	# mto(a0,a1) = |C_a1| + 2 * |E_a1|
	# 中间是修改的
	# aco1 = 2 * sum(map(len, a1.values())) + len(a1)
	# aco2 = 2 * sum(map(len, a2.values())) + len(a2)
	# 中间是修改的

	# 原本代码中的
	aco1=sum(map(len,a1.values()))+len(a1)
	aco2=sum(map(len,a2.values()))+len(a2)
	# 原本代码中的
	# print(aco1)
	# print(aco2)

	a2a = 1 - float(mto) / (float(aco1) + float(aco2))
	print("end")
	print("a2a的值: " + str(a2a))
	print(dis_sum)
	if dis_sum != 0:
		for key, value in res.items():
			res[key] = f"{value / dis_sum:.5f}"
	else:
		for key, value in res.items():
			res[key] = 0

	print(res)
	return res, a2a, dis_sum

if __name__ == '__main__':
	# if len(sys.argv) != 3:
	# 	print("Usage: python git_diff_file_modify_aquire.py <rsf1_file> <rsf2_file> ")
	# 	sys.exit(1)
	rsf1_rsf = r"../sema_results/libuv-1.49.0/libuv-1.49.0_NamedClusters.rsf"
	rsf2_rsf = r"../sema_results/libuv-1.49.1/libuv-1.49.1_NamedClusters.rsf"
	# output_file = sys.argv[3]
	a2a(rsf1_rsf,rsf2_rsf)



