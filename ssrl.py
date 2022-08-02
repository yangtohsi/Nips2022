#!/usr/bin/env python3

from typing import Dict
from datautil import DatasetLoader
from features import FeatureBinarizer
import argparse
import time
import os
# from rulelist import RuleListClassifier
from pyroaring import BitMap
import itertools
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class subproblem_solver():
    '''
    This class solve the problem
    |A\cap_{e\in S} e|-|B\cap_{e\in S} e|-lambda_0*(|C\cap_{e\in S} e|+lambda_1|S|)
    which equals to
    |\bar{B}\cup_{e\in S} \bar{e}|+lambda_0*|\bar{C}\cap_{e\in S} e|-|\bar{A}\cap_{e\in S} \bar{e}|-lambda_0*lambda_1|S|
    '''
    def __init__(self,bitmap=None,patience_ratio=0.1,cc = 10):
        self.bitmap = bitmap
        self.last_rule = []
        self.current_rule = []
        self.final_rule = []
        self.patience_ratio = patience_ratio
        self.full_set = None
        self.cc = cc
        if self.bitmap is None:
            self.full_feature_list = []
            self.feature_list = []
            self.max_patience = None
        else:
            self.full_feature_list = list(self.bitmap.keys())
            self.feature_list = list(self.bitmap.keys())
            self.max_patience = self.patience_ratio*len(self.feature_list)
    def load_bitmap(self,bitmap,full_set):
        self.bitmap = bitmap
        self.full_feature_list = list(self.bitmap.keys())
        self.feature_list = list(self.bitmap.keys())
        self.max_patience = self.patience_ratio*len(self.full_feature_list)
        self.full_set = full_set
    def initial_solution_1(self,A,B,C,lambda_0,lambda_1,M=16,K=3):
        init_rule = []
        bit_list = [self.bitmap[feature] for feature in init_rule]
        if len(bit_list)<1:
            bit_intersect = self.full_set
        else:
            bit_intersect = BitMap.intersection(*bit_list)
        for idx in range(len(self.feature_list)):
            max_ratio = -1e+15
            for feature in self.feature_list:
                if feature in init_rule:
                    continue
                gain_A = BitMap.intersection(A,bit_intersect).difference_cardinality(self.bitmap[feature])
                gain_B = BitMap.intersection(B,bit_intersect).difference_cardinality(self.bitmap[feature])
                gain_C = BitMap.intersection(C,bit_intersect).difference_cardinality(self.bitmap[feature])
                
                ratio = (gain_B+lambda_0*gain_C+self.cc)/(gain_A+lambda_0*lambda_1+self.cc)
                if ratio>max_ratio:
                    max_ratio = ratio
                    best_feature = feature
            if max_ratio>1+1e-5:
                init_rule.append(best_feature)
                bit_intersect = bit_intersect.intersection(self.bitmap[best_feature])
            else:
                break
        return init_rule
        if len(init_rule)<K:
            return init_rule
        else:
            possible_sbusets = list(itertools.combinations(init_rule, K))
            max_gain = -1e+15
            for k_subset in possible_sbusets:
                bit_list = [self.bitmap[feature] for feature in k_subset]
                if len(bit_list)<1:
                    bit_intersect = self.full_set
                else:
                    bit_intersect = BitMap.intersection(*bit_list)
                gain = A.intersection_cardinality(bit_intersect)\
                        -B.intersection_cardinality(bit_intersect)\
                        -lambda_0*C.intersection_cardinality(bit_intersect)
                if gain>max_gain:
                    max_gain = gain
                    best_subset = k_subset
            return [*best_subset]
    def initial_solution(self,A,B,C,lambda_0,lambda_1):
        N = len(self.feature_list)
        M = 16
        K = 3
        nn = max(round(N/M),5)
        best_gain = -1e+15
        for ii in range(300):
            init_rule = []
            A_bit_intersect = A
            B_bit_intersect = B
            C_bit_intersect = C
            for idx in range(M):
                max_gain = -1e+15
                check_list = np.random.permutation(N)[:nn].tolist()
                for idx_1 in check_list:
                    feature = self.feature_list[idx_1]
                    gain = -BitMap.difference_cardinality(A_bit_intersect,self.bitmap[feature])\
                            +BitMap.difference_cardinality(B_bit_intersect,self.bitmap[feature])\
                            +lambda_0*BitMap.difference_cardinality(C_bit_intersect,self.bitmap[feature])
                    if gain>max_gain:
                        max_gain = gain
                        best_feature = feature
                if max_gain>lambda_0*lambda_1:  
                    init_rule.append(best_feature)
                    A_bit_intersect = BitMap.intersection(A_bit_intersect,self.bitmap[best_feature])
                    B_bit_intersect = BitMap.intersection(B_bit_intersect,self.bitmap[best_feature])
                    C_bit_intersect = BitMap.intersection(C_bit_intersect,self.bitmap[best_feature])
            gain_1 = len(A_bit_intersect)-len(B_bit_intersect)\
                    -lambda_0*(len(C_bit_intersect)+lambda_1*len(init_rule))
            if gain_1>best_gain:
                best_gain = gain_1
                best_init = init_rule
        return best_init,best_gain
    
    def local_search(self,A,B,C,lambda_0,lambda_1):
        insert_improve = True
        delete_improve = True
        replace_improve = True
        while insert_improve or delete_improve or replace_improve:
            insert_improve = False
            delete_improve = False
            replace_improve = False
            # Insert
            bit_list = [self.bitmap[feature_1] for feature_1 in self.final_rule]
            bit_list.append(self.full_set)
            bit_intersect = BitMap.intersection(*bit_list)
            A_bit_intersect = BitMap.intersection(A,bit_intersect)
            B_bit_intersect = BitMap.intersection(B,bit_intersect)
            C_bit_intersect = BitMap.intersection(C,bit_intersect)
            for feature in self.feature_list:
                if feature in self.final_rule:
                    continue
                gain = -A_bit_intersect.difference_cardinality(self.bitmap[feature])\
                       +B_bit_intersect.difference_cardinality(self.bitmap[feature])\
                       +lambda_0*C_bit_intersect.difference_cardinality(self.bitmap[feature])\
                       -lambda_0*lambda_1
                if gain>0:
                    self.final_rule.append(feature)
                    A_bit_intersect = BitMap.intersection(A_bit_intersect,self.bitmap[feature])
                    B_bit_intersect = BitMap.intersection(B_bit_intersect,self.bitmap[feature])
                    C_bit_intersect = BitMap.intersection(C_bit_intersect,self.bitmap[feature])
                    insert_improve = True
            # replace
            for feature in self.final_rule:
                bit_list = [self.bitmap[feature_1] for feature_1 in self.final_rule if feature_1!=feature]
                bit_list.append(self.full_set)
                bit_intersect = BitMap.intersection(*bit_list)
                A_bit_intersect = BitMap.intersection(A,bit_intersect)
                B_bit_intersect = BitMap.intersection(B,bit_intersect)
                C_bit_intersect = BitMap.intersection(C,bit_intersect)
                ori_gain = -A_bit_intersect.difference_cardinality(self.bitmap[feature])\
                           +B_bit_intersect.difference_cardinality(self.bitmap[feature])\
                           +lambda_0*C_bit_intersect.difference_cardinality(self.bitmap[feature])\
                           -lambda_0*lambda_1
                max_gain = -1e+15
                for feature_2 in self.feature_list:
                    if feature_2 in self.final_rule:
                        continue
                    gain = -A_bit_intersect.difference_cardinality(self.bitmap[feature_2])\
                           +B_bit_intersect.difference_cardinality(self.bitmap[feature_2])\
                           +lambda_0*C_bit_intersect.difference_cardinality(self.bitmap[feature_2])\
                           -lambda_0*lambda_1
                    if gain>max_gain:
                        best_replace = feature_2
                        max_gain = gain
                if max_gain > ori_gain:
                    self.final_rule.remove(feature)
                    self.final_rule.append(best_replace)
                    replace_improve = True
            # delete
            if len(self.final_rule)>1:
                for feature in self.final_rule:
                    bit_list = [self.bitmap[feature_1] for feature_1 in self.final_rule if feature_1!=feature]
                    bit_list.append(self.full_set)
                    bit_intersect = BitMap.intersection(*bit_list)
                    gain = BitMap.intersection(A,bit_intersect).difference_cardinality(self.bitmap[feature])\
                           -BitMap.intersection(B,bit_intersect).difference_cardinality(self.bitmap[feature])\
                           -lambda_0*BitMap.intersection(C,bit_intersect).difference_cardinality(self.bitmap[feature])\
                           +lambda_0*lambda_1
                    if gain>0:
                        self.final_rule.remove(feature)
                        if len(self.final_rule)<2:
                            break
                        delete_improve = True
    

    def get_permutation(self,A,B,C,lambda_0,lambda_1):
        permutation = {}
        unused_feature = [feature for feature in self.feature_list if feature not in self.last_rule]
        # rand perm
        score_A_list_1 = []
        score_O_list_1 = []
        score_ratio_list_1 = [] 
        for feature in self.last_rule:
            bit_list = [self.bitmap[feature_1] for feature_1 in self.last_rule if feature_1!=feature]
            if len(bit_list)<1:
                bit_intersect = self.full_set
            else:
                bit_intersect = BitMap.intersection(*bit_list)
            gain_A = BitMap.intersection(A,bit_intersect).difference_cardinality(self.bitmap[feature])
            gain_B_C = BitMap.intersection(B,bit_intersect).difference_cardinality(self.bitmap[feature])\
                       +lambda_0*BitMap.intersection(C,bit_intersect).difference_cardinality(self.bitmap[feature])
            gain_len = lambda_0*lambda_1
            score_A_list_1.append((feature,gain_B_C))
            gain_O = gain_B_C - gain_A - gain_len
            score_O_list_1.append((feature,gain_O))
            score_ratio_list_1.append((feature,(gain_B_C+self.cc)/(gain_A+gain_len+self.cc)))
        score_A_list_2 = []
        score_O_list_2 = []
        score_ratio_list_2 = [] 
        bit_list = [self.bitmap[feature_1] for feature_1 in self.last_rule]
        if len(bit_list)<1:
            bit_intersect = self.full_set
        else:
            bit_intersect = BitMap.intersection(*bit_list)
        A_bit_intersect = BitMap.intersection(A,bit_intersect)
        B_bit_intersect = BitMap.intersection(B,bit_intersect)
        C_bit_intersect = BitMap.intersection(C,bit_intersect)
        for feature in unused_feature:
            gain_A = A_bit_intersect.difference_cardinality(self.bitmap[feature])
            gain_B_C = B_bit_intersect.difference_cardinality(self.bitmap[feature])\
                           +lambda_0*C_bit_intersect.difference_cardinality(self.bitmap[feature])
            gain_len = lambda_0*lambda_1
            score_A_list_2.append((feature,gain_B_C))
            gain_O = gain_B_C-gain_A-gain_len
            score_O_list_2.append((feature,gain_O))

            score_ratio_list_2.append((feature,(gain_B_C+self.cc)/(gain_A+gain_len+self.cc)))
        score_A_list_1 = sorted(score_A_list_1,key=lambda x:x[-1],reverse=True)
        score_A_list_2 = sorted(score_A_list_2,key=lambda x:x[-1],reverse=True)
        score_O_list_1 = sorted(score_O_list_1,key=lambda x:x[-1],reverse=True)
        score_O_list_2 = sorted(score_O_list_2,key=lambda x:x[-1],reverse=True)
        score_ratio_list_1 = sorted(score_ratio_list_1,key=lambda x:x[-1],reverse=True)
        score_ratio_list_2 = sorted(score_ratio_list_2,key=lambda x:x[-1],reverse=True)
        K = len(self.last_rule)
        N = len(self.feature_list)
        perm_1 = np.random.permutation(K).tolist()
        perm_2 = np.random.permutation(N-K).tolist()
        permutation['random'] = [[self.last_rule[idx] for idx in perm_1],[unused_feature[idx] for idx in perm_2]]
        permutation['gain_A'] = [[item[0] for item in score_A_list_1],[item[0] for item in score_A_list_2]]
        permutation['gain_O'] = [[item[0] for item in score_O_list_1],[item[0] for item in score_O_list_2]] 
        permutation['gain_ratio'] = [[item[0] for item in score_ratio_list_1],[item[0] for item in score_ratio_list_2]] 
        return permutation
 
    
#     def pre_search(self,A,B,C,lambda_0,lambda_1,lower_bound=None):
#         if lower_bound is None:
#             lower_bound = -1e+15
#         self.feature_list = []
#         one_layer_best = []
#         for feature in self.full_feature_list:
#             gain_lower_bound = A.intersection_cardinality(self.bitmap[feature])-lambda_0*lambda_1
#             full_gain = gain_lower_bound-B.intersection_cardinality(self.bitmap[feature])\
#                     -lambda_0*C.intersection_cardinality(self.bitmap[feature])
#             if gain_lower_bound>lower_bound:
#                 self.feature_list.append(feature)
#             if full_gain>lower_bound:
#                 lower_bound = full_gain
#                 one_layer_best = [feature]
#         N = len(self.feature_list)
#         two_layer_best = None
#         for idx_1 in range(N-1):
#             for idx_2 in range(idx_1,N):
#                 join_bitmap = BitMap.intersection(self.bitmap[self.feature_list[idx_1]],\
#                                                   self.bitmap[self.feature_list[idx_2]])
#                 gain_lower_bound = A.intersection_cardinality(join_bitmap)-lambda_0*lambda_1
#                 full_gain = gain_lower_bound-B.intersection_cardinality(join_bitmap)\
#                         -lambda_0*C.intersection_cardinality(join_bitmap)
#                 if full_gain>lower_bound:
#                     lower_bound = full_gain
#                     two_layer_best = [self.feature_list[idx_1],self.feature_list[idx_2]]
#         if two_layer_best is None:
#             return one_layer_best
#         else:
#             return two_layer_best
    
        
    def solve(self,A,B,C,lambda_0,lambda_1,lower_bound=None):
        """
        This function maximize 
               |A \cap x_1 \cap x_2 ...|-|B \cap x_1 \cap x_2 ...|-lambda_0(|C \cap x_1 \cap x_2 ...|+lambda_1 |X|)
        using modmod method.
        """
        self.feature_list = []
        if lower_bound is not None:
            for feature in self.full_feature_list:
                if A.intersection_cardinality(self.bitmap[feature])-lambda_0*lambda_1>lower_bound:
                    self.feature_list.append(feature)
        else:
            self.feature_list.extend(self.full_feature_list)
        if len(self.feature_list)<1:
            return self.full_feature_list[0],-1e+15
#         self.last_rule,_ = self.initial_solution(A,B,C,lambda_0,lambda_1)
        self.last_rule = self.initial_solution_1(A,B,C,lambda_0,lambda_1)
        self.current_rule = []
        self.final_rule = []
        converge = False
        N = len(self.feature_list)
        unused_feature = [feature for feature in self.feature_list if feature not in self.last_rule]
        patience = 0
        iter_1 = 0
        while not converge:
            iter_1+=1
            score_list_1 = []
            score_list_2 = []
            # compute 2 lower bounds of first term
            final_gain = -1e+15
            permutation = self.get_permutation(A,B,C,lambda_0,lambda_1)
            for method_name,perm in permutation.items():
#                 if method_name == 'random':
#                     continue
#                 if method_name == 'gain_A':
#                     continue
# #                 if method_name == 'gain_O':
# #                     continue
#                 if method_name == 'gain_ratio':
#                     continue
                perm_1,perm_2 = perm
                pre_include_B = B
                pre_include_C = C
    #             for next_feature in self.last_rule:
                for next_feature in perm_1:
                    score_B = BitMap.difference_cardinality(pre_include_B,self.bitmap[next_feature])
                    score_C = BitMap.difference_cardinality(pre_include_C,self.bitmap[next_feature])
                    pre_include_B = BitMap.intersection(pre_include_B,self.bitmap[next_feature])
                    pre_include_C = BitMap.intersection(pre_include_C,self.bitmap[next_feature])
                    pre_include_A_1 = [self.bitmap[feature_1] for feature_1 in self.last_rule if feature_1!=next_feature]
                    pre_include_A_1.append(A)
                    low_bound_A1 = -BitMap.difference_cardinality(BitMap.intersection(*pre_include_A_1),\
                                                                  self.bitmap[next_feature])
                    pre_include_A_2 = [self.bitmap[feature_1] for feature_1 in self.feature_list\
                                       if feature_1!=next_feature]
                    pre_include_A_2.append(A)
                    low_bound_A2 = -BitMap.difference_cardinality(BitMap.intersection(*pre_include_A_2),\
                                                                  self.bitmap[next_feature])

                    score_1 = low_bound_A1+score_B+lambda_0*score_C-lambda_0*lambda_1
                    score_2 = low_bound_A2+score_B+lambda_0*score_C-lambda_0*lambda_1

                    score_list_1.append((next_feature,score_1))
                    score_list_2.append((next_feature,score_2))

                pre_include_A_2 = [self.bitmap[feature_1] for feature_1 in self.last_rule]
                pre_include_A_2.append(A)
                pre_include_A_2_bit = BitMap.intersection(*pre_include_A_2)
                for next_feature in perm_2:
                    score_B = BitMap.difference_cardinality(pre_include_B,self.bitmap[next_feature])
                    score_C = BitMap.difference_cardinality(pre_include_C,self.bitmap[next_feature])
                    pre_include_B = BitMap.intersection(pre_include_B,self.bitmap[next_feature])
                    pre_include_C = BitMap.intersection(pre_include_C,self.bitmap[next_feature])
                    low_bound_A1 = -BitMap.difference_cardinality(A,self.bitmap[next_feature])
                    low_bound_A2 = -BitMap.difference_cardinality(pre_include_A_2_bit,\
                                                                  self.bitmap[next_feature])
                    score_1 = low_bound_A1+score_B+lambda_0*score_C-lambda_0*lambda_1
                    score_2 = low_bound_A2+score_B+lambda_0*score_C-lambda_0*lambda_1

                    score_list_1.append((next_feature,score_1))
                    score_list_2.append((next_feature,score_2))

                solution_1 = []
                solution_2 = []
                score_list_1 = sorted(score_list_1,key=lambda x:x[-1],reverse=True)
                score_list_2 = sorted(score_list_2,key=lambda x:x[-1],reverse=True)
                for feature,score in score_list_1:
                    if score > 0:
                        solution_1.append(feature)
                    else:
                        break
                for feature,score in score_list_2:
                    if score > 0:
                        solution_2.append(feature)
                    else:
                        break
                if len(solution_1)<1:
                    solution_1.append(score_list_1[0][0])
                if len(solution_2)<1:
                    solution_2.append(score_list_2[0][0])
                bit_list_1 = [self.bitmap[feature] for feature in solution_1]
                bit_intersect_1 = BitMap.intersection(*bit_list_1)
                bit_list_2 = [self.bitmap[feature] for feature in solution_2]
                bit_intersect_2 = BitMap.intersection(*bit_list_2)

                gain_1 = A.intersection_cardinality(bit_intersect_1)-B.intersection_cardinality(bit_intersect_1)\
                    -lambda_0*C.intersection_cardinality(bit_intersect_1)-lambda_0*lambda_1*len(solution_1)
                gain_2 = A.intersection_cardinality(bit_intersect_2)-B.intersection_cardinality(bit_intersect_2)\
                    -lambda_0*C.intersection_cardinality(bit_intersect_2)-lambda_0*lambda_1*len(solution_2)
                if gain_1>gain_2:
                    current_rule = solution_1
                    current_gain = gain_1
                else:
                    current_rule = solution_2
                    current_gain = gain_2
                if current_gain>final_gain:
                    final_gain = current_gain
                    self.final_rule = current_rule
            if set(self.final_rule) == set(self.last_rule):
                patience += 1
                if patience > 20:
                    converge = True
            else:
                patience = 0
                self.last_rule = self.final_rule
                unused_feature = [feature for feature in self.feature_list if feature not in self.last_rule]
        self.local_search(A,B,C,lambda_0,lambda_1)

        bit_list = [self.bitmap[feature] for feature in self.final_rule]
        if len(bit_list)>0:
            bit_intersect = BitMap.intersection(*bit_list)
            final_gain = A.intersection_cardinality(bit_intersect)-B.intersection_cardinality(bit_intersect)\
                    -lambda_0*C.intersection_cardinality(bit_intersect)-lambda_0*lambda_1*len(self.final_rule)
        else:
            final_gain = len(A)-len(B)-lambda_0*len(C)
        return self.final_rule,final_gain
        
class submodular_rule_list_learner():
    def __init__(self,lambda_1,distorted_step=10,cc=10,use_multi_pool=False):
        self.lambda_1 = lambda_1
        self.default_rule_weight = 1
        self.current_rulelist = []
        self.last_rulelist = []
        self.final_rulelist = []
        self.full_set = None
        self.subproblem_solver = subproblem_solver(cc=cc)
        self.distorted_step = distorted_step
        self.use_multi_pool = use_multi_pool
    def get_bitmap(self,dataset,label_column,feature_columns):
        self.feature_bitmap_dict = {}
        self.label_bitmap_dict = {}
        for column in feature_columns:
            one_hot_feature = pd.get_dummies(dataset[[column]].astype('str'),prefix_sep='=').reset_index(drop=True)
            for one_hot_column in list(one_hot_feature.columns):
                self.feature_bitmap_dict[one_hot_column] = BitMap(one_hot_feature[one_hot_feature[one_hot_column]==1].index)

        
        label_series = dataset[label_column].reset_index(drop=True)
        self.full_set = BitMap(label_series.index)
        
        self.subproblem_solver.load_bitmap(self.feature_bitmap_dict,self.full_set)
        default_rules = []
        for item in dataset[label_column].unique():
            self.label_bitmap_dict[item] = BitMap(label_series[label_series==item].index)
            rule_info = {}
            rule_info['name'] = '=>'+item
            rule_info['length'] = 1
            rule_info['covered'] = self.full_set
            rule_info['miss_classified'] = BitMap(label_series[label_series!=item].index)
            rule_info['correct_classified'] = BitMap(label_series[label_series==item].index)
            default_rules.append(rule_info)
        return default_rules
    
    def get_cover_and_cl(self,rule_list=None):
        if rule_list is None:
            rule_list = self.current_rulelist
        covered = BitMap([])
        correct_cls = BitMap([])
        for rule in rule_list:
            correct_cls = BitMap.union(BitMap.difference(rule['correct_classified'],covered),correct_cls)
            covered = BitMap.union(covered,rule['covered'])
            
        return covered,correct_cls
    
    def evaluate_effct_on_default(self,rule,mode,default_rule):
#         tic_2 = time.time()
        if mode == 0:
            if rule in self.last_rulelist:
                cover_list = [rule_1['covered'] for rule_1 in self.last_rulelist if rule_1 != rule]
                cover_list.append(default_rule['miss_classified'])
                effect = BitMap.difference_cardinality(rule['covered'],BitMap.union(*cover_list))
            else:
                effect = BitMap.difference_cardinality(rule['covered'],default_rule['miss_classified'])
        if mode == 1:
            if rule in self.last_rulelist:
                effect = 0.0
            else:
                cover_list = [rule_1['covered'] for rule_1 in self.last_rulelist]
                cover_list.append(default_rule['miss_classified'])
                effect = BitMap.difference_cardinality(rule['covered'],BitMap.union(*cover_list))
#         tic_3 = time.time()
#         print('2nd step '+str(tic_3-tic_2))
        return effect+self.lambda_1*rule['length']
        
    def evaluate_ori_gain(self,default_rule,rule_list=None,lambda_1=None,default_rule_weight=None):
        if rule_list is None:
            rule_list = self.final_rulelist
        gain = -1e+15
        if len(rule_list)<1:
            return gain
        if lambda_1 is None:
            lambda_1 = self.lambda_1
        if default_rule_weight is None:
            default_rule_weight = self.default_rule_weight
        prefix_rule = []
        gain_rule_len = 0.0
        for rule in rule_list:
            gain_rule_len += rule['length']
        pre_covered, pre_correct_cls = self.get_cover_and_cl(rule_list)

        gain_prefix_cls = len(pre_correct_cls)
        gain_default_cls = BitMap.union_cardinality(pre_covered,default_rule['miss_classified'])
        
        gain = gain_prefix_cls-self.default_rule_weight*(gain_default_cls+lambda_1*gain_rule_len)
        
        return gain
    
    def insert_ele_w_position(self,mode,default_rule,position):
        optimal_rulelist = []
        max_gain_insert = 0
        N = len(self.current_rulelist)
        pre_rulelist = [self.current_rulelist[ii] for ii in range(position)]
        pre_covered, pre_correct_cls = self.get_cover_and_cl(pre_rulelist)
        pos_rulelist = [self.current_rulelist[ii] for ii in range(position,N)]
        post_covered, post_correct_cls = self.get_cover_and_cl(pos_rulelist)

        for rule in self.last_rulelist:
            if rule in self.current_rulelist:
                continue
            ele_cls_gain = BitMap.difference_cardinality(rule['correct_classified'],pre_covered)
            ele_cls_pos_gain = BitMap.difference_cardinality(BitMap.intersection(post_correct_cls,\
                                                            rule['covered']),pre_covered)
            gain_rule_len = self.evaluate_effct_on_default(rule,mode,default_rule)
            gain = ele_cls_gain-ele_cls_pos_gain-self.default_rule_weight*gain_rule_len
            if gain>max_gain_insert:
                max_gain_insert = gain
                optimal_rulelist = []
                optimal_rulelist.extend(pre_rulelist)
                optimal_rulelist.append(rule)
                optimal_rulelist.extend(pos_rulelist)

        

        for label,label_bitmap in self.label_bitmap_dict.items():
            if position<N:
                if label == self.current_rulelist[position]['label_name']:
                    continue
            if position >= N:
                if ('=>'+label) == default_rule['name']:
#                     print(label)
                    continue
#                 if idx>0:
#                     if label == self.current_rulelist[idx-1]['label_name']:
#                         continue
            A = BitMap.difference(label_bitmap,pre_covered)
#                 if len(A)<self.default_rule_weight*self.lambda_1:
#                     continue
            if len(A)-self.default_rule_weight*self.lambda_1<max_gain_insert:
                continue
            B = BitMap.difference(post_correct_cls,pre_covered)
#                 print(B)
            if mode == 0:
                C = default_rule['correct_classified']
                rule,gain = self.subproblem_solver.solve(A=A,B=B,C=C,lambda_0=self.default_rule_weight,\
                                                         lambda_1=self.lambda_1,\
                                                         lower_bound=max_gain_insert)
            else:
                cover_list = [rule_1['covered'] for rule_1 in self.last_rulelist]
                cover_list.append(default_rule['miss_classified'])
                C = BitMap.difference(self.full_set,BitMap.union(*cover_list))
                rule,gain = self.subproblem_solver.solve(A=A,B=B,C=C,lambda_0=self.default_rule_weight,\
                                                         lambda_1=self.lambda_1,\
                                                         lower_bound=max_gain_insert)
#                 print((rule,gain))
#                 print(label)
#                 print(self.default_rule_weight)
            if gain > max_gain_insert:
                max_gain_insert = gain
                rule_info = {}
                rule_info['name'] = '&'.join(sorted(rule))+'=>'+label
                bit_list = [self.feature_bitmap_dict[feature] for feature in rule]
                rule_info['covered'] = BitMap.intersection(*bit_list)
                rule_info['correct_classified'] = BitMap.intersection(rule_info['covered'],self.label_bitmap_dict[label])
                rule_info['length'] = len(rule)
                rule_info['label_name'] = label
                optimal_rulelist = []
                optimal_rulelist.extend(pre_rulelist)
                optimal_rulelist.append(rule_info)
                optimal_rulelist.extend(pos_rulelist)
        return max_gain_insert,optimal_rulelist
    def insert_element(self,mode,default_rule):
        improved = False
        N = len(self.current_rulelist)
        if self.use_multi_pool:
            arg_list = [(mode,default_rule,idx) for idx in range(N+1)]
            pool = Pool()
            insert_results = pool.starmap(self.insert_ele_w_position, arg_list)
            pool.close()
            pool.join()
        else:
            insert_results = []
            for ii in range(N+1):
                insert_results.append(self.insert_ele_w_position(mode,default_rule,ii))
#         print(insert_results)
        sort_results = sorted(insert_results,key=lambda x:x[0],reverse=True)
        if sort_results[0][0]>1e-5:
            self.current_rulelist = sort_results[0][1]
            improved = True
#         assert 1==0       
        return improved
    
    def approx_replace_element(self,mode,default_rule):
        improved = False
        N = len(self.current_rulelist)
        optimal_rulelist = []
        for idx in range(N):
            pre_rulelist = [self.current_rulelist[ii] for ii in range(idx)]
            pre_covered, pre_correct_cls = self.get_cover_and_cl(pre_rulelist)
            pos_rulelist = [self.current_rulelist[ii] for ii in range(idx+1,N)]
            post_covered, post_correct_cls = self.get_cover_and_cl(pos_rulelist)

            ele_cls_gain = BitMap.difference_cardinality(self.current_rulelist[idx]['correct_classified'],\
                                                         pre_covered)
            ele_cls_pos_gain = BitMap.difference_cardinality(BitMap.intersection(post_correct_cls,\
                                                            self.current_rulelist[idx]['covered']),pre_covered)
            gain_rule_len = self.evaluate_effct_on_default(self.current_rulelist[idx],mode,default_rule)
            base_gain = ele_cls_gain-ele_cls_pos_gain-self.default_rule_weight*gain_rule_len
            
            max_gain_replace = base_gain
            
            for rule in self.last_rulelist:
                if rule in self.current_rulelist:
                    continue
                ele_cls_gain = BitMap.difference_cardinality(rule['correct_classified'],pre_covered)
                ele_cls_pos_gain = BitMap.difference_cardinality(BitMap.intersection(post_correct_cls,\
                                                                rule['covered']),pre_covered)
                gain_rule_len = self.evaluate_effct_on_default(rule,mode,default_rule)
                gain = ele_cls_gain-ele_cls_pos_gain-self.default_rule_weight*gain_rule_len
                if gain>max_gain_replace:
                    max_gain_insert = gain
                    optimal_rulelist = []
                    optimal_rulelist.extend(pre_rulelist)
                    optimal_rulelist.append(rule)
                    optimal_rulelist.extend(pos_rulelist)
            
            for label,label_bitmap in self.label_bitmap_dict.items():
#                 if idx<N:
#                     if label == self.current_rulelist[idx]['label_name']:
#                         continue
#                 if idx>0:
#                     if label == self.current_rulelist[idx-1]['label_name']:
#                         continue
                A = BitMap.difference(label_bitmap,pre_covered)
#                 if len(A)<self.default_rule_weight*self.lambda_1:
#                     continue
                if len(A)-self.default_rule_weight*self.lambda_1<max_gain_replace:
                    continue
                B = BitMap.difference(post_correct_cls,pre_covered)
#                 print(B)
                if mode == 0:
                    C = default_rule['correct_classified']
                    rule,gain = self.subproblem_solver.solve(A=A,B=B,C=C,lambda_0=self.default_rule_weight,\
                                                             lambda_1=self.lambda_1,\
                                                             lower_bound=max_gain_replace)
                else:
                    cover_list = [rule_1['covered'] for rule_1 in self.last_rulelist]
                    cover_list.append(default_rule['miss_classified'])
                    C = BitMap.difference(self.full_set,BitMap.union(*cover_list))
                    rule,gain = self.subproblem_solver.solve(A=A,B=B,C=C,lambda_0=self.default_rule_weight,\
                                                             lambda_1=self.lambda_1,\
                                                             lower_bound=max_gain_replace)
#                 print((rule,gain))
#                 print(label)
#                 print(self.default_rule_weight)
                if gain > max_gain_replace:
                    max_gain_insert = gain
                    rule_info = {}
                    rule_info['name'] = '&'.join(sorted(rule))+'=>'+label
                    bit_list = [self.feature_bitmap_dict[feature] for feature in rule]
                    rule_info['covered'] = BitMap.intersection(*bit_list)
                    rule_info['correct_classified'] = BitMap.intersection(rule_info['covered'],self.label_bitmap_dict[label])
                    rule_info['length'] = len(rule)
                    rule_info['label_name'] = label
                    optimal_rulelist = []
                    optimal_rulelist.extend(pre_rulelist)
                    optimal_rulelist.append(rule_info)
                    optimal_rulelist.extend(pos_rulelist)
                    
            if max_gain_replace>base_gain+1e-5:
                self.current_rulelist = optimal_rulelist
                improved = True    
        return improved
    
    def approx_delete_element(self,mode,default_rule):
        improved = False
        all_removed = False
        while not all_removed:
            N = len(self.current_rulelist)
            optimal_rulelist = []
            min_gain_delete = 0
            for idx in range(N):
                pre_rulelist = [self.current_rulelist[ii] for ii in range(idx)]
                pre_covered, pre_correct_cls = self.get_cover_and_cl(pre_rulelist)
                pos_rulelist = [self.current_rulelist[ii] for ii in range(idx+1,N)]
                post_covered, post_correct_cls = self.get_cover_and_cl(pos_rulelist)

                ele_cls_gain = BitMap.difference_cardinality(self.current_rulelist[idx]['correct_classified'],\
                                                             pre_covered)
                ele_cls_pos_gain = BitMap.difference_cardinality(BitMap.intersection(post_correct_cls,\
                                                                self.current_rulelist[idx]['covered']),pre_covered)
                gain_rule_len = self.evaluate_effct_on_default(self.current_rulelist[idx],mode,default_rule)
                base_gain = ele_cls_gain-ele_cls_pos_gain-self.default_rule_weight*gain_rule_len

                if base_gain<min_gain_delete:
                    min_gain_delete = base_gain
                    optimal_rulelist = []
                    optimal_rulelist.extend(pre_rulelist)
                    optimal_rulelist.extend(pos_rulelist)

            if min_gain_delete<0:
                self.current_rulelist = optimal_rulelist
                improved = True
            else:
                all_removed = True
        return improved
    
    def approx_exchange_elements(self,default_rule):
        improved = False
        covered, correct_cls = self.get_cover_and_cl(self.current_rulelist)
        current_gain = len(correct_cls)
        N = len(self.current_rulelist)
        for idx_i in range(N-1):
            for idx_j in range(1,N):
                new_rulelist = [self.current_rulelist[idx_1] for idx_1 in range(idx_i)]
                new_rulelist.append(self.current_rulelist[idx_j])
                new_rulelist.extend([self.current_rulelist[idx_1] for idx_1 in range(idx_i+1,idx_j)])
                new_rulelist.append(self.current_rulelist[idx_i])
                new_rulelist.extend([self.current_rulelist[idx_1] for idx_1 in range(idx_j+1,N)])
                covered, correct_cls = self.get_cover_and_cl(self.current_rulelist)
                new_gain = len(correct_cls)
                if new_gain>current_gain+1e-5:
                    current_gain = new_gain
                    self.final_rulelist = new_rulelist
                    improved = True
        return improved
    
    def delete_element(self,default_rule):
        improved = False
        all_removed = False
        while not all_removed:
            current_gain = self.evaluate_ori_gain(default_rule)
            max_gain = current_gain
            N = len(self.final_rulelist)
            for idx in range(N):
                new_rulelist = [self.final_rulelist[idx_1] for idx_1 in range(idx)]
                new_rulelist.extend([self.final_rulelist[idx_1] for idx_1 in range(idx+1,N)])
                new_gain = self.evaluate_ori_gain(default_rule,rule_list=new_rulelist)
                if new_gain>max_gain+1e-5:
                    max_gain = new_gain
                    optimal_rulelist = new_rulelist
            if max_gain>current_gain+1e-5:
                self.final_rulelist = optimal_rulelist
                improved = True
            else:
                all_removed = True
                
        return improved
    
    def replace_element(self,default_rule):
        improved = False
        N = len(self.final_rulelist)
        optimal_rulelist = []
        for idx in range(N):
            
            pre_rulelist = [self.final_rulelist[ii] for ii in range(idx)]
            pre_covered, pre_correct_cls = self.get_cover_and_cl(pre_rulelist)
            pos_rulelist = [self.final_rulelist[ii] for ii in range(idx+1,N)]
            post_covered, post_correct_cls = self.get_cover_and_cl(pos_rulelist)
            cover_on_default = BitMap.union(pre_covered,post_covered,default_rule['miss_classified'])
            
            old_ele_cls_gain = BitMap.difference_cardinality(self.final_rulelist[idx]['correct_classified'],\
                                                             pre_covered)
            old_ele_cls_pos_gain = BitMap.difference_cardinality(BitMap.intersection(post_correct_cls,\
                                                        self.final_rulelist[idx]['covered']),pre_covered)
            old_gain_rule_len = BitMap.difference_cardinality(self.final_rulelist[idx]['covered'],\
                                            cover_on_default)+self.lambda_1*self.final_rulelist[idx]['length']
            old_gain = old_ele_cls_gain-old_ele_cls_pos_gain-self.default_rule_weight*old_gain_rule_len
            max_gain = old_gain
            for label,label_bitmap in self.label_bitmap_dict.items():
                A = BitMap.difference(label_bitmap,pre_covered)
                if len(A)<self.default_rule_weight*self.lambda_1:
                    continue
                B = BitMap.difference(post_correct_cls,pre_covered)
                C = BitMap.difference(self.full_set,cover_on_default)
                rule,gain = self.subproblem_solver.solve(A=A,B=B,C=C,lambda_0=self.default_rule_weight,\
                                                         lambda_1=self.lambda_1,\
                                                         lower_bound = max_gain)
                if gain>max_gain:
                    max_gain = gain
                    new_rule = []
                    new_rule.extend(rule)
                    new_rule_label = label
                
            if max_gain > old_gain+1e-5:
                rule = {}
                rule['name'] = '&'.join(sorted(new_rule))+'=>'+new_rule_label
                bit_list = [self.feature_bitmap_dict[feature] for feature in new_rule]
                rule['covered'] = BitMap.intersection(*bit_list)
                rule['correct_classified'] = BitMap.intersection(rule['covered'],self.label_bitmap_dict[new_rule_label])
                rule['length'] = len(new_rule)
                rule['label_name'] = new_rule_label
                optimal_rulelist = []
                optimal_rulelist.extend(pre_rulelist)
                optimal_rulelist.append(rule)
                optimal_rulelist.extend(pos_rulelist)
                self.final_rulelist = optimal_rulelist
                improved = True

        return improved
    
    def exchange_elements(self,default_rule):
        improved = False
        current_gain = self.evaluate_ori_gain(default_rule)
        N = len(self.final_rulelist)
        for idx_i in range(N-1):
            for idx_j in range(1,N):
                new_rulelist = [self.final_rulelist[idx_1] for idx_1 in range(idx_i)]
                new_rulelist.append(self.final_rulelist[idx_j])
                new_rulelist.extend([self.final_rulelist[idx_1] for idx_1 in range(idx_i+1,idx_j)])
                new_rulelist.append(self.final_rulelist[idx_i])
                new_rulelist.extend([self.final_rulelist[idx_1] for idx_1 in range(idx_j+1,N)])
                new_gain = self.evaluate_ori_gain(default_rule,rule_list=new_rulelist)
                if new_gain>current_gain+1e-5:
                    current_gain = new_gain
                    self.final_rulelist = new_rulelist
                    improved = True
        return improved
    
    
    def sovle_approx_func(self,default_rule):
#         self.last_rulelist = []
#         for feature,feature_bitmap in self.feature_bitmap_dict.items():
# #             if np.random.rand(1)>0.8:
# #                 continue
#             max_cover = 0
#             for label,label_bitmap in self.label_bitmap_dict.items():
#                 cover = label_bitmap.intersection_cardinality(feature_bitmap)
#                 if cover>max_cover:
#                     max_cover = cover
#                     rule_label = label
# #             if '=>'+rule_label == default_rule['name']:
# #                 continue
#             rule = {}
#             rule['name'] = feature+'=>'+rule_label
#             rule['covered'] = feature_bitmap
#             rule['correct_classified'] = BitMap.intersection(rule['covered'],self.label_bitmap_dict[rule_label])
#             rule['length'] = 1
#             rule['label_name'] = rule_label
#             self.last_rulelist.append(rule)

        self.last_rulelist = []
        iter_1 = 0
        for feature,feature_bitmap in self.feature_bitmap_dict.items():
            for label,label_bitmap in self.label_bitmap_dict.items():
                rule = {}
                rule['name'] = feature+'=>'+label
                rule['covered'] = feature_bitmap
                rule['correct_classified'] = BitMap.intersection(rule['covered'],label_bitmap)
                rule['length'] = 1
                rule['label_name'] = label
                self.last_rulelist.append(rule)
            iter_1 += 1
            if iter_1>3:
                break
#         self.last_rulelist = []
#         label = default_rule['name'].split('=>')[1]
#         for feature,feature_bitmap in self.feature_bitmap_dict.items():
#             rule = {}
#             rule['name'] = feature+'=>'+label
#             rule['covered'] = feature_bitmap
#             rule['correct_classified'] = BitMap.intersection(rule['covered'],default_rule['correct_classified'])
#             rule['length'] = 1
#             rule['label_name'] = label
#             self.last_rulelist.append(rule)
        converge = False
        iter_1 = 0
        while not converge:
            iter_1 += 1
            max_gain = -1e+15
            for mode in reversed(range(2)):
                self.current_rulelist = []
                for ii in range(self.distorted_step):
                    self.default_rule_weight = (1-1/self.distorted_step)**(-self.distorted_step+ii+1)
                    improve = self.insert_element(mode,default_rule)
                    if not improve:
                        break
                self.default_rule_weight = 1

                for ii in range(30):                    
                    insert_improve = self.insert_element(mode,default_rule)
                    if not insert_improve:
                        break
                gain = self.evaluate_ori_gain(default_rule,rule_list=self.current_rulelist)
#                 print('===========Rules after Iter '+str(iter_1)+' mode '+str(mode)+' ======================')
#                 print([rule['name'] for rule in self.current_rulelist])
#                 print('gain='+str(gain))
                if gain>max_gain:
                    self.final_rulelist = self.current_rulelist
                    max_gain = gain
#             gain, rule_list = self.local_search(rule_dict,default_rule)
            last_rulelist = [rule['name'] for rule in self.last_rulelist]
            final_rulelist = [rule['name'] for rule in self.final_rulelist]
            if set(last_rulelist) == set(final_rulelist):
                converge = True
            else:
                self.last_rulelist = self.final_rulelist
            if iter_1>10:
                break
        return max_gain, self.final_rulelist
    
    
    def local_search(self,default_rule):
        delete_improved = True
        exchange_improved = True
        replace_improved = True
        while delete_improved or exchange_improved or replace_improved:
            exchange_improved = self.exchange_elements(default_rule)
#             print('===========2. rules after exchange step======================')
#             print([rule['name'] for rule in self.final_rulelist])
            replace_improved = self.replace_element(default_rule)
#             print('===========3. rules after replace step======================')
#             print([rule['name'] for rule in self.final_rulelist])
            delete_improved = self.delete_element(default_rule)
#             print('===========4. rules after delete step======================')
#             print([rule['name'] for rule in self.final_rulelist])
            gain = self.evaluate_ori_gain(default_rule)
            cl_gain = self.evaluate_ori_gain(default_rule,lambda_1=0)
#             print('===>Iter '+str(iter_1)+': '+str(len(self.final_rulelist))+' rules got, current gain is '+str(gain)+', and classification gain is '+str(cl_gain)+', and classfication accuracy is '+str(confidence))
#             print([rule['name'] for rule in self.final_rulelist])
#             print('exchange_improved='+str(exchange_improved))
#             print('replace_improved='+str(replace_improved))
#             print('delete_improved='+str(delete_improved))
            
        return gain, self.final_rulelist
    
    def train(self,dataset,label_column,feature_columns):
        M = dataset.shape[0]
#         print('Prepare the BitMap(s) of the rules')
        default_rulelist = self.get_bitmap(dataset,label_column,feature_columns)
        best_gain = -1e+15
        cl_gain_best = -1e+15
        best_rulelist = []
        for default_rule in default_rulelist:
#             if default_rule['name'] == '=>positive':
#                 continue
#             if default_rule['name'] == '=>M':
#                 continue
            if default_rule['name'] == '=>1':
                continue
#             print('Learning with default rule set to '+default_rule['name'])
#             for repeat in range(2):
            self.sovle_approx_func(default_rule)
            gain, rule_list = self.local_search(default_rule)
#             self.last_rulelist = self.final_rulelist
            self.default_rule_weight = 1
            cl_gain = self.evaluate_ori_gain(default_rule,lambda_1=0)
#             print(self.default_rule_weight)
#             print('Classification acc is '+str(1+cl_gain/M))
            if cl_gain > best_gain:
                best_gain = cl_gain
                best_rulelist = []
                best_rulelist.extend(rule_list)
                best_rulelist.append(default_rule)
                cl_gain_best = self.evaluate_ori_gain(default_rule,rule_list=best_rulelist,lambda_1=0)
            elif (abs(cl_gain - best_gain)<1e-4) & (len(rule_list)<len(best_rulelist)-1):
                best_gain = cl_gain
                best_rulelist = []
                best_rulelist.extend(rule_list)
                best_rulelist.append(default_rule)
                cl_gain_best = self.evaluate_ori_gain(default_rule,rule_list=best_rulelist,lambda_1=0)
#             print(best_gain)
        self.final_rulelist = best_rulelist
#         print('=================================================================')
#         print('The training classification acc is '+str(1+cl_gain_best/M))
#         print('The learnt final rule is ')
#         print([rule['name'] for rule in self.final_rulelist])
#         return best_rulelist
            
    
    def analysis_rules(self,rule_list=None):
        if rule_list is None:
            rule_list = [rule['name'] for rule in self.final_rulelist]
        final_rulelist = []
        label_list = []
        for rule in rule_list:
            result_cond = []
            condition, label = rule.split('=>')
            label_list.append(label)
            if condition == '':
                final_rulelist.append([])
            else:
                final_rulelist.append(condition.split('&'))
        return final_rulelist,label_list
    
    def predict(self,data):
        rulelist,label_list = self.analysis_rules()
#         print('Predict using following rule list')
#         print(rulelist)
        data_new = pd.DataFrame({col:str(col)+'=' for col in list(data.columns)}, index=data.index)\
                            +data.astype('str')
        result_df = pd.DataFrame(np.zeros((data.shape[0],1)), index=data.index)
        for idx, row in data_new.iterrows():
            for idx_r in range(len(rulelist)):
                if set(rulelist[idx_r]).issubset(row):
#                     print(row)
#                     print(rulelist[idx_r])
                    result_df.loc[idx]=label_list[idx_r]
                    break
                
        return result_df
