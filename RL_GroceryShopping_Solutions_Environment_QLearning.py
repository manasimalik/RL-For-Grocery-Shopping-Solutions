
# coding: utf-8

# In[1]:

import itertools
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import copy
import random


# In[2]:

#Data
MRPs=dict()
MRPs[0]=20
MRPs[1]=100
MRPs[2]=50
MRPs[3]=50
MRPs[4]=100
MRPs[5]=60
MRPs[6]=35
MRPs[7]=216
MRPs[8]=27
MRPs[9]=130
MRPs[10]=160
MRPs[11]=89
MRPs[12]=73
MRPs[13]=27
MRPs[14]=185
MRPs[15]=249
MRPs[16]=199
MRPs[17]=46
MRPs[18]=55
MRPs[19]=99
'''
MRPs['chocolate']=20
MRPs['shampoo']=100
MRPs['chips']=50
MRPs['pulses']=50
MRPs['apples']=100
MRPs['grapes']=60
MRPs['pepsi']=35
MRPs['bournvita']=216
MRPs['bread']=27
MRPs['jam']=130
MRPs['buscuits']=160
MRPs['butter']=89
MRPs['toothpaste']=73
MRPs['eggs']=27
MRPs['flour']=185
MRPs['peanut_butter']=249
MRPs['cheese']=199
MRPs['milk']=46
MRPs['yoghurt']=55
MRPs['muffins']=99
'''

no_shops=4
no_items=2

actions=range(no_shops)

reward_buying=50

#Bernoulli variable for each shop
bernoulli=np.random.rand(no_shops,no_items)

#Price bias for each shop
bias=np.random.normal(0,5,no_shops)

#Distance Matrix
a = np.random.uniform(1,10,(no_shops,no_shops))
distance_matrix = np.tril(a) + np.tril(a, -1).T
np.fill_diagonal(distance_matrix,0)

print 'Distance between shops'
print distance_matrix


# In[3]:

def price_penalty(next_state):
    scaling=0.1
    shop=next_state[0]
    next_status=next_state[1]
    price=0
    for item_no in range(len(next_status)):
        if next_state[1][item_no]:
            price=price+np.random.normal(MRPs[item_no]+bias[shop],1)
        
    return -price*scaling


# In[4]:

def distance_penalty(distance):
    return -distance


# In[5]:

def availability_in_shop(current_state,next_state):
    old_status=current_state[1]
    new_status=next_state[1]
    next_shop=next_state[0]
    
    prob=1
    
    for item_no in range(len(old_status)):
        if old_status[item_no]==0:
            if new_status[item_no]==0:
                prob=prob*(1-bernoulli[next_shop][item_no])
            else:
                prob=prob*bernoulli[next_shop][item_no]
        else:
            if new_status[item_no]==0:
                prob=0
                
    return prob


# In[6]:

def M(shop_b,shop_a):
    temp=sum(sum(np.triu(distance_matrix)))
    temp2=(temp-distance_matrix[shop_b,shop_a])/((no_shops-1)*temp)
    return temp2
    


# In[7]:

#Creating State Space
state_space=[]
all_possible_buying_statuses=list(itertools.product([0, 1], repeat=no_items))

for shop_no in range(no_shops):
    for buying_status in all_possible_buying_statuses:
        state=(shop_no,buying_status)
        state_space.append(state)

print 'State Space Size'
print len(state_space)


# In[8]:

#Defining Transition Probabilities and Rewards
P=dict()
R=dict()

actions=range(no_shops)
for current_state, action, next_state in list(itertools.product(state_space,actions,state_space)):
    if current_state[1]==tuple(np.ones(no_items)):
        P[(current_state,action,next_state)]=0
        R[(current_state,action,next_state)]=None
        continue
    
    if action == current_state[0]: #action==current_shop
        if next_state[0]!=current_state[0]:
            P[(current_state,action,next_state)]=0
            R[(current_state,action,next_state)]=None
        else:
            P[(current_state,action,next_state)]=availability_in_shop(current_state,next_state)
            bought_items=sum(next_state[1])
            if bought_items>0:
                R[(current_state,action,next_state)]=bought_items*reward_buying+distance_penalty(distance_matrix[current_state[0]][next_state[0]])+price_penalty(next_state)
            else:
                R[(current_state,action,next_state)]=distance_penalty(distance_matrix[current_state[0]][next_state[0]])
    else:
        if next_state[0]==action:
            P[(current_state,action,next_state)]=0.9*availability_in_shop(current_state,next_state)
            bought_items=sum(next_state[1])
            if bought_items>0:
                R[(current_state,action,next_state)]=bought_items*reward_buying+distance_penalty(distance_matrix[current_state[0]][next_state[0]])+price_penalty(next_state)
            else:
                R[(current_state,action,next_state)]=distance_penalty(distance_matrix[current_state[0]][next_state[0]])
        else:
            P[(current_state,action,next_state)]=0.1*availability_in_shop(current_state,next_state)*M(next_state[0],current_state[0])
            bought_items=sum(next_state[1])
            if bought_items>0:
                R[(current_state,action,next_state)]=bought_items*reward_buying+distance_penalty(distance_matrix[current_state[0]][action]+distance_matrix[action][next_state[0]])+price_penalty(next_state)
            else:
                R[(current_state,action,next_state)]=distance_penalty(distance_matrix[current_state[0]][action]+distance_matrix[action][next_state[0]])
    
    print "Next transition"
    print current_state, action, next_state, P[(current_state,action,next_state)],R[(current_state,action,next_state)]


# In[9]:

def takeaction(current_state,action):
    global P
    global R
    global state_space
    r = random.random()
    for next_state in state_space:
        if(r<=0):
            break
        r-=P[(current_state,action,next_state)]
    return next_state, R[(current_state,action,next_state)]


# In[10]:

def e_greedy(e,Q_s):
    x=random.randrange(1,11)
    if x<=e*10:
        return random.randrange(no_shops)
    else:
        return np.argmax(Q_s)


# In[11]:

def q_learning(no_episodes,no_steps,alpha,discount,epsilon):
    Q=dict()
    Rewards=[]
    
    for e in range(no_episodes):
        S=random.choice(state_space)
        step=0
        Episode_Reward=0
        
        while(step<no_steps):
            if S not in Q.keys():
                Q[S]=np.zeros(no_shops).astype(int)
            if S[1]==tuple(np.ones(no_items)):
                break
            
            A=e_greedy(epsilon,Q[S])
            S_,r=takeaction(S,A)
            
            if S_ not in Q.keys():
                Q[S_]=np.zeros(no_shops).astype(int)
            
            A_=np.argmax(Q[S_])
            
            if r==None:
                r=0
            Q[S][A]=Q[S][A]+alpha*(r+discount*Q[S_][A_]-Q[S][A])
            S=S_
            step+=1
            Episode_Reward+=r
        
        Rewards.append(Episode_Reward)   
    return Q,Rewards
        


# In[12]:

result_Q,Rewards=q_learning(1000,200,0.1,0.9,0.5) 


# In[13]:

print "Best Policy for each state"
print ""
best_policy=dict()
for state in result_Q:
    best_action=np.argmax(result_Q[state])
    status=state[1]
    if status==tuple(np.ones(no_items)):
        best_action="End"
    best_policy[state]=best_action
    print "State",state, "Best Action",best_action
    

print ""
print "Rewards"
plt.plot(Rewards)
plt.xlabel('No. of episodes')
plt.ylabel('Reward in that Episode')
plt.show()


# In[14]:

Rewards_Qpolicy=[]
Rewards_RandomPolicy=[]
for test_no in range(100):
    print "Test no.",test_no
    S=random.choice(state_space)
    S_random=S
    step=0
    r=0
    r_random=0
    
    while(S[1]!=tuple(np.ones(no_items)) and step<200):
        A=best_policy[S]
        S_,rew=takeaction(S,A)
        S=S_
        rew=R[(S,A,S_)]
        #print S,A,S_,rew
        if rew==None:
            break
        r+=rew
        step+=1
    Rewards_Qpolicy.append(r)
    
    step=0
    while(S_random[1]!=tuple(np.ones(no_items)) and step<200):
        A_random=random.choice(actions)
        S_random_,rew=takeaction(S_random,A_random)
        S_random=S_random_
        rew=R[(S_random,A_random,S_random_)]
        #print S_random,A_random,S_random_,rew
        
        if rew==None:
            break
        r_random+=rew
        step+1
    Rewards_RandomPolicy.append(r_random)
            
print ""
print "Rewards "
plt.plot(Rewards_Qpolicy,color='r',label='With Q learning Policy')
plt.plot(Rewards_RandomPolicy,color='g',label='With Random Policy')
plt.xlabel('Test No.')
plt.ylabel('Rewards')
plt.legend()
plt.show()


# In[ ]:



