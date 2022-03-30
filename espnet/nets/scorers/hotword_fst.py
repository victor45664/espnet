"""Ngram lm implement."""

from abc import ABC

import torch

from espnet.nets.scorer_interface import BatchScorerInterface


from typing import List
class Arc(object):
    def __init__(self,idx,w,out_node,cost):
        self.idx=idx
        self.w=w   #该弧所代表的 词或者bpe
        self.out_node=out_node   #该弧指向的节点
        self.cost=cost   #经过该弧的得分
    def __str__(self):
        return "w:"+str(self.w)+" outnode:"+str(self.out_node.idx)


class Node(object):
    def __init__(self, idx, out_arcs: List[Arc],return_cost=0):
        self.idx=idx
        self.out_arcs=out_arcs   #从该节点出发的弧
        self.return_cost=return_cost    #从该节点出发回到根节点的arc的分数，这个arc就省略了
    def __str__(self):
        outstr=""
        outstr+="Nodeidx:"+str(self.idx)
        for i in range(len(self.out_arcs)):
            outstr+=" Arc"+str(self.out_arcs[i].idx)+":"+str(self.out_arcs[i])+",cost:"+str(self.out_arcs[i].cost)
        return outstr

    def next_node(self,w):  #从这个节点出发的所有arc中，找到匹配的arc并且返回目的地节点
        #使用这函数之前需要判断arc.w是否存在，参考self.__contains__
        for arc in self.out_arcs:
            if arc.w==w:
                return arc.out_node
        print("Warning:"+w+" is not in this node")
        return None

    def __contains__(self, key):
        for arc in self.out_arcs:
            if arc.w==key:
                return True
        return False   #表示从该node中出发的arc中是否包含这个key

class hotword_FST(ABC):
    #hotword_list should only contain ids
    def __init__(self,hotwords,award=1):
        self.award=award  #hotword的奖励,改变fusion weight 应该和这个效果一样。
        self.all_nodes=[Node(0,[])]  #起始节点,起始节点的return_cost为0
        self.all_arcs=[]  #起始节点
        hotwords.sort(key=lambda x: len(x),reverse=True)  #需要从长到短添加 hotword，非常关键
        for hw in hotwords:
            self.__add_hotword(hw)


    def __add_hotword(self,hw):
        #这个函数不可以从外部访问，因为hotword的添加必须要遵循从长到短添加
        node_p=self.all_nodes[0]   #节点指针 首先指向起点
        for i in range(len(hw)-1):
            w=hw[i]
            if w in node_p:
                node_p=node_p.next_node(w)
            else:
                newnode=Node(len(self.all_nodes), [],-(i+1)*self.award)
                self.all_nodes.append(newnode)
                node_p.out_arcs.append(Arc(len(self.all_arcs),w,newnode,self.award))
                self.all_arcs.append(node_p.out_arcs[-1])
                node_p=node_p.next_node(w)    #上面已经添加了，这一步到达新添加的节点
        w = hw[-1]
        if w in node_p:  #这说明当前hw是之前一个hw的子词，因此当前节点
            node_p=node_p.next_node(w)
            node_p.return_cost=0       # 因为已经匹配完毕一个子词，因此不做惩罚
        else:
            newreturnarc=Arc(len(self.all_arcs),w,self.all_nodes[0],self.award) #指向起点的arc
            node_p.out_arcs.append(newreturnarc)
            self.all_arcs.append(newreturnarc)

    def init_state(self,):
        return 0

    def score(self,state,w):
        #state是nodes的编号
        #w是当前输入符号
        node_p=self.all_nodes[state]
        if w in node_p:   #判断从node_p出发的arc中是否存在w
            node_p=node_p.next_node(w)   #存在就行走
        else:
            node_p = self.all_nodes[0]  #返回初始节点
        new_state=node_p.idx
        out_w=[]
        out_w_cost=[]
        for arc in node_p.out_arcs:
            out_w.append(arc.w)  #当前节点的所有输出弧
            out_w_cost.append(arc.cost)

        return new_state, [out_w, out_w_cost], node_p.return_cost  # 当前节点的输出弧度和对应的cost，和当前节点的返回cost
        # 值得注意的是对于hotwordFST来说所有arc的cost都是一样的














class HwFSTFullScorer(BatchScorerInterface):

    def __init__(self,hw_list,num_of_token):
        #从path中读取hw_list
        #['▁FANNY', '▁CAME', '▁WALKING', '▁IN', '▁UPON', '▁HER', '▁ONE', '▁DAY', '▁LOOKING', '▁AS', '▁IF', '▁SHE', '▁BROUGHT', '▁TI', 'D', 'ING', 'S', '▁OF', '▁SUCH', '▁GREAT', '▁JOY', '▁THAT', '▁SHE', '▁HARDLY', '▁KNEW', '▁HOW', '▁TO', '▁TELL', '▁THEM']
        #[4999, 4341, 123, 1732, 8, 94, 24, 46, 118, 394, 23, 61, 29, 400, 517, 30, 17, 3, 5, 122, 117, 817, 11, 29, 981, 287, 121, 6, 207, 63]
        #hw_list=[]  #请注意，这里要转成ids的形式
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

