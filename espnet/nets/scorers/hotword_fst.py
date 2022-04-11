"""Ngram lm implement."""

from abc import ABC

import torch
#
from espnet.nets.scorer_interface import BatchScorerInterface

def gen_const(i,total_length):
    return 1

def gen_linear(i,total_length):
    return i

from typing import List




class Node(object):
    def __init__(self, idx, w,
                 out_nodes=None, fail_cost=0,
                 out_nodes_cost=None, fail_node= None, end_node=False,
                 father_node=None):
        if out_nodes_cost is None:
            out_nodes_cost = []
        if out_nodes is None:
            out_nodes = []
        self.idx=idx              #该节点的idx
        self.w=w               #当前节点的符号

        self.out_nodes=out_nodes   #从该节点出发，可以到达的节点
        self.out_nodes_cost=out_nodes_cost    #从该节点出发，到其它节点的cost

        self.is_end_node=end_node   #是否是重点节点

        self.fail_cost=fail_cost    #前往fail节点的cost
        self.fail_node=fail_node       #fail节点，默认指向root节点
        self.temp_total_score=0      #匹配这个单词到达这个节点的全部分数，这个在score的时候不会用，只是用来step2，计算fail 的cost
        self.father_node=father_node   #指向父节点
    def next_node(self,w):  #从这个节点出发的所有arc中，找到匹配的arc并且返回目的地节点
        #使用这函数之前需要判断arc.w是否存在，参考self.__contains__
        for i in range(len(self.out_nodes)):
            node=self.out_nodes[i]
            if node.w==w:
                return True,node,self.out_nodes_cost[i]   #匹配成功，返回下一个节点和cost

        return False,self.fail_node,self.fail_cost   #匹配失败，返回失败节点和cost

    def __contains__(self, w):
        #输出节点中是否存在w
        for i in range(len(self.out_nodes)):
            node=self.out_nodes[i]
            if node.w==w:
                return True  #匹配成功
        return False


    def __str__(self):
        outstr=str(self.idx)+":"+str(self.w)+"--->"
        for node in self.out_nodes:
            outstr+=str(node.idx)+":"+str(node.w)+" "
        outstr += "  Fail-->"+str(self.fail_node.idx)+":"+str(self.fail_node.w)+" Fcost"+str(self.fail_cost)
        return outstr

class hotword_FST(ABC):
    #hotword_list should only contain ids
    def __init__(self,hotwords,award=1,gen_score_f=gen_const):
        self.gen_score_f=gen_score_f #每匹配到一个符号，的分数增量
        self.award=award  #hotword的奖励,改变fusion weight 应该和这个效果一样。
        self.all_nodes=[Node(0, None)]  #起始节点,起始节点的return_cost为0,而且这个节点不代表任何符号，根节点没有fail_node 和 父节点

        #hotwords.sort(key=lambda x: len(x),reverse=True)  #需要从长到短添加 hotword，非常关键
        #step1 :add all hotwords and build a initial FST
        for hw in hotwords:
            self.__add_hotword(hw)
        #at this point every word have been added but all the fail path is pointing to root node
        #step2: re-route fail path
        self.__BFS_re_route(self.all_nodes[0])

        #step3: shorten the fail path
        #the FST is still usable without step3 but it will be slower
        #too difficult skip

        pass

    def score(self,state,w):
        #state是nodes的编号
        #w是当前输入符号
        node_p=self.all_nodes[state]

        #process next state
        ismatch,next_node,cost =node_p.next_node(w)
        if ismatch:   #匹配
            node_p=next_node
        else:
            node_p,cost = self.__go_through_fail_path(node_p,w)  #当前节点没有匹配，在fail_path 中搜索
                                                                #this function is very expensive and will be called twice
                                                                #some optimization can be done here?
        new_state=node_p.idx  #next state done
        # process score
        local_w=[]
        local_w_cost=[]
        for i in range(len(node_p.out_nodes)):
            local_w.append(node_p.out_nodes[i].w)     #这里score的是直接的
            local_w_cost.append(node_p.out_nodes_cost[i])

        fail_w,fail_w_cost,fail_cost=self.__score_fail_path(node_p)

        #merge to score
        for i in range(len(fail_w)):
            w=fail_w[i]
            if w in local_w:
                pass    #本地匹配的优先
            else:
                local_w.append(w)
                local_w_cost.append(fail_w_cost[i])
        return new_state, [local_w, local_w_cost], fail_cost  # 当前节点的输出弧度和对应的cost，和当前节点的返回cost
        # 值得注意的是对于hotwordFST来说所有arc的cost都是一样的


    def __score_fail_path(self,node):
        #这个函数会遍历整个fail_path，找到所有可能的next token并计算分数，
        fail_w=[]
        fail_w_cost=[]
        total_fail_cost=0
        node_p=node
        while(True):
            total_fail_cost+=node_p.fail_cost
            node_p=node_p.fail_node
            if node_p==None:
                break
            for i in range(len(node_p.out_nodes)):
                if node_p.out_nodes[i].w not in fail_w:
                    fail_w.append(node_p.out_nodes[i].w)
                    fail_w_cost.append(total_fail_cost+node_p.out_nodes_cost[i])

        return fail_w,fail_w_cost,total_fail_cost


    def __go_through_fail_path(self,node,w):
        node_p=node
        total_cost=0
        while(node_p.fail_node!=None):
            total_cost+=node_p.fail_cost
            if w in node_p.fail_node:
                ismatch,next_node,cost=node_p.fail_node.next_node(w)
                return next_node,cost+total_cost
            else:
                node_p=node_p.fail_node
        return self.all_nodes[0],total_cost
    def __BFS_re_route(self,root_node: Node):
        children_node=[root_node]
        while(len(children_node)!=0):
            temp_children_node=[]
            for node in children_node:
                self.__re_route_fail_path(node)
                temp_children_node+=node.out_nodes
            children_node=temp_children_node



    # def __re_route_fail_path(self,node: Node):
    #     w=node.w
    #     node_p=node.father_node
    #
    #     if node.idx==0 or node_p.idx==0:
    #         return  #father node or this node is root node  don't do anything
    #     if node.fail_cost == 0 and node.fail_node.idx == 0:  # fail_cost是0并且返回root节点，说明这个节点是一个结束节点，不应该被更改
    #         return
    #     while(node_p.fail_node!=None):
    #
    #         if w in node_p.fail_node:
    #             ismatch,next_node,cost=node_p.fail_node.next_node(w)
    #             node.fail_node=next_node
    #             node.fail_cost=next_node.temp_total_score-node.temp_total_score  #fail cost 需要被改变，这样顺着fail path回到部分匹配单词的时候分数才会正确
    #             return
    #         else:
    #             node_p=node_p.fail_node
    #     return #no match do nothing

    def __re_route_fail_path(self,node: Node):
        w=node.w
        node_p=node.father_node

        if node.idx==0 or node_p.idx==0:
            return  #father node or this node is root node  don't do anything
        if node.fail_cost == 0 and node.fail_node.idx == 0:  # fail_cost是0并且返回root节点，说明这个节点是一个结束节点，不应该被更改
            return
        _,_,total_return_cost=node_p.next_node(w)
        total_return_cost=-total_return_cost
        while(node_p.fail_node!=None):
            total_return_cost+=node_p.fail_cost
            if w in node_p.fail_node:
                ismatch,next_node,cost=node_p.fail_node.next_node(w)
                node.fail_node=next_node
                node.fail_cost=total_return_cost+cost  #fail cost 需要被改变，这样顺着fail path回到部分匹配单词的时候分数才会正确
                return
            else:
                node_p=node_p.fail_node
        node.fail_cost=total_return_cost
        return #no match do nothing




    def __add_hotword(self,hw):
        node_p=self.all_nodes[0]   #节点指针 首先指向起点
        score_at_each_arc=self.__gen_score(len(hw))
        for i in range(len(hw)-1):
            w=hw[i]
            match,next_node,cost=node_p.next_node(w)
            if match: #当前节点有一个输出节点与符号w一样。
                node_p=next_node
            else:
                newnode= Node(len(self.all_nodes),w,
                              fail_node=self.all_nodes[0],fail_cost=-sum(score_at_each_arc[:i + 1]),father_node=node_p) #添加新的节点，默认fail回到root节点
                newnode.temp_total_score=sum(score_at_each_arc[:i + 1]) #当前词到到当前节点累积的分数，用来在step2中计算fail的cost
                self.all_nodes.append(newnode)
                node_p.out_nodes.append(newnode)
                node_p.out_nodes_cost.append(score_at_each_arc[i])
                node_p=newnode
        w = hw[-1]
        match, next_node, cost = node_p.next_node(w)
        if match:  #这说明当前hw是之前一个hw的子词，因此当前节点,已经完成了匹配，fail不应该惩罚
            next_node.fail_cost=0
        else:
            newnode = Node(len(self.all_nodes), w,
                           fail_node=self.all_nodes[0],fail_cost=0,father_node=node_p)  # 添加新的节点，已经匹配完毕，默认fail回到root节点,不需要fail_cost
            newnode.temp_total_score = sum(score_at_each_arc)
            self.all_nodes.append(newnode)
            node_p.out_nodes.append(newnode)
            node_p.out_nodes_cost.append(score_at_each_arc[-1])

    def init_state(self,):
        return 0



    def __gen_score(self,length):
        #generate socre using self.
        score=[]
        for i in range(length):
            score.append(self.gen_score_f(i,length))

        divid=sum(score)/(self.award*length)

        for i in range(length):
            score[i]=score[i]/divid   #normalize score so that the sum of score equals to self.award*length
        return score



class HwFSTFullScorer(BatchScorerInterface):

    def __init__(self,hw_list,num_of_token):
        #从path中读取hw_list
        self.fst=hotword_FST(hw_list)
        self.num_of_token=num_of_token  #bpe的总数

    """Fullscorer for Hw fst."""
    def init_state(self, x):
        """Initialize tmp state."""
        return 0


    def score(self, y, state, x):

        newstate,sparse_score,return_cost=self.fst.score(state,int(y[-1])) #请确保hotword list中的ids与这里的一致

        score = torch.ones((self.num_of_token), dtype=x.dtype, device=y.device)*return_cost  #大部分分数都是returncost

        for i in range(len(sparse_score[0])):
            w,w_cost=sparse_score[0][i],sparse_score[1][i]
            score[w]=w_cost
        return score,newstate


# if __name__ == "__main__":


#     # #Case 1
#     # input_str=['g','g','a','b','c','e','f','g','g','g','g']  #input sequence
#     # hFST=hotword_FST([
#     #     ['a','b','c','d'],
#     #     ['e','f','g']
#     # ])  #hotword

#     # #Case 2
#     # input_str='gggabcdefgggg'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefa",
#     #     "defg",
#     # ])  #hotword total score 4

#     # #Case 3
#     # input_str='gggabcdefgggg'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefa",
#     #     "cdef",
#     # ])  #hotword total score 4

#     #Case 4
#     # input_str='gggabcdefcgggg'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefg",
#     #     "cdef",
#     #     "efg",
#     # ])  #hotword 4

#     #Case 5
#     # input_str='zzabcdefghlfzz'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefghi",
#     #     "cdefghi",
#     #     "efghlf",
#     # ])  #hotword 6

#     # #Case 6
#     # input_str=['z', 'z', 'z', 'a', 'b', 'c', 'd', 'e', 'z', 'z', 'z', 'z']
#     # hFST=hotword_FST([['a', 'b', 'c', 'd', 'f'],
#     #                   ['b', 'c', 'd', 'f'],
#     #                   ['c', 'd', 'f'],
#     #                   ['d', 'e'],
#     #                   ])  #hotword 2


#     # #Case 7
#     # input_str='zzzabcdefzzzz'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefg",
#     #     "abc",
#     #     "fgz",
#     # ])  #hotword 3
#     #Case 8
#     # input_str='zzzabcdefgzzzz'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefge",
#     #     "abc",
#     #     "fgz",
#     # ])  #hotword 3
#     #Case 9
#     input_str='zzzabcdefzzzz'  #input sequence
#     hFST=hotword_FST([
#         "abcdefghi",
#         "abc",
#         "abcde",
#         "abcdefg",
#         "abcdefa",
#     ])  #hotword 3


#     # input_str='zzzabcdefghigzzzz'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefghia",
#     #     "efghig",
#     #     "fghiga",
#     #     "ghig",
#     # ])  #hotword 3
#     # input_str='zzzabcdefghigzzzz'  #input sequence
#     # hFST=hotword_FST([
#     #     "abcdefgh",
#     #     "abc",
#     #     "def",
#     # ])  #hotword 3

#     state=hFST.init_state()
#     total_score=0
#     for i in range(len(input_str)-1):
#         print(input_str[:i+2])
#         state,output_score,return_cost=hFST.score(state,input_str[i])

#         if input_str[i+1] in output_score[0]:
#             total_score+=output_score[1][output_score[0].index(input_str[i+1])]
#         else:
#             total_score+=return_cost
#         print(output_score)
#         print(return_cost)
#         print("total_score",total_score)
#         print("=============")
#         gg=1