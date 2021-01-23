#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

#NUM_MACRO = 1
#SECTOR_PER_MACRO = 3
#NUM_PICO_PER_SECTOR = 2
#NUM_UE_PER_SECTOR = 30
NUM_PICO = 6   #PBS的个数
NUM_UE = 90   #用户的个数
INTERSITE_DIS = 500   # meters, intersite distance #站间距离
MIN_DIS_MP = 75       # meters, minimum distance between macrocell and picocell #M基站和P基站之间的最短距离
MIN_DIS_PP = 40       # meters, minimum distance between picocell and picocell #P基站与P基站之间的最短距离
MIN_DIS_MU = 35       #M基站和用户之间的最短距离  用于计算M基站和用户之间的路径损失
MIN_DIS_PU = 10       #P基站和用户之间的最短距离  用于计算P基站和用户之间的路径损失

SUB_BW = 180          # kHz, subchannel bandwidth  #子信道的带宽
NUM_SUB = 100         # number of subchannels      #子信道的个数
REUSE = 1             # frequency reuse for picocells  #标志着P基站频率是否可用
#dBm是一个表示功率绝对值的值（也可以认为是以1mW功率为基准的一个比值），计算公式为：10log（功率值/1mw）。
#dBi是表示天线功率增益的量，两者都是一个相对值，但参考基准不一样。dBi的参考基准为全方向性天线。
P_MAX_MACRO = 46      # dBm, maximum transmit power of macrocell  #相当于40w
P_MAX_PICO = 30       # dBm, maximum transmit power of picocell   #相当于1w
ANT_GAIN_MACRO = 15   # dBi, macrocell antenna gain
ANT_GAIN_PICO = 5     # dBi, picocell antenna gain
SHADOW_FADING = 8    # dB,  shadow fading
#dB是一个表征相对值的值，纯粹的比值，只表示两个量的相对大小关系，没有单位，
# 当考虑甲的功率相比于乙功率大或小多少个dB时，按下面计算公式
# 10log（甲功率/乙功率），如果采用两者的电压比计算，要用20log（甲电压/乙电压）。

# path_loss_macro = 128.1 + 37.6 lg(R) dB
# path_loss_pico = 140.7 + 36.7 lg(R) dB
# feasible pico_cell_position 
RANDOM_SEED = 77
TOTAL_SCHEDULING_INT = 60-2 #调度次数，每过1秒调度一次，超过该阈值一个情景就结束
NOISE_DENSITY = -174   #themal noise, -174dBm/Hz
MEGA = 1000000.0    #用来把数据传输率的单位转换为Mbs

PICO_DEPLOY = './pico_deploy/pico_pos.txt'
UE_TRACES = './user_traces/'

all_pico_deploy = np.loadtxt(PICO_DEPLOY)
#pico_deploy_idx = np.random.randint(len(all_pico_deploy))
pico_deploy_idx = 0


class Environment:
    """初始化的只有基站位置，以及用户位置数据"""
    def __init__(self, all_user_pos, random_seed=RANDOM_SEED):
        np.random.seed(random_seed)         #保证生成的随机数不变，状态之间的转换概率是固定的。
        self.macrocell = np.zeros((1,2))    # macrocell position #坐标为(0,0)
        self.picocells = all_pico_deploy.reshape((6,2)) #把从文本中读出的p基站坐标转化为6行，2列的数组（表示6个P基站的位置）
        self.all_user_pos = all_user_pos    #加载已经训练好的用户位置数据
        #轨迹的索引
        self.trace_idx = 0
        self.user_pos = self.all_user_pos[self.trace_idx]  #当从第一个轨迹中的用户位置数据开始
        self.scheduling_ptr = 0  #表示调度的次数
        
        self.current_user_pos = np.zeros((90,2))   #当前的用户位置都初始化为(0,0)
        self.K = 10     # subchannel number of frequency reuse-1 for picocells
        #根据后面的代码可知，这里的association定义为(7,90)更好，（0,3)=1表示第四个用户被分配到M基站
        #这np.zeros((90,7))赋值没有用，最后在scheduling_and_association又被重新赋值了。
        self.association = np.zeros((90,7))  # UE-BS association array 0 for macrocell, 1-6 for picocell

        #保证这90个用户会一定分配给这7个基站中的一个，且每个用户在任何时候都只能连接到一个基站上去
        #总共有7个基站，索引为0的为MBS，索引为1-6的为PBS

        #用来进行调度和关联，用来产生下一个状态


        #输入一个用户分配方案和共享的子信道个数，输出用户的状态（就是位置，即使信道增益），和奖励等信息
    """该函数类似于maze函数中的step函数"""
    def scheduling_and_association(self, association, num_shared):
        #当前用户的位置数据，在第一次取当前所有用户位置数据和每次的坐标（x,y），所以这里的间隔是2
        #这个状态也是强化学习中的初始状态
        """先计算出每个用户和所有基站之间的信息增益"""
        current_user_pos = self.user_pos[:, self.scheduling_pt
                                         r*2:(self.scheduling_ptr+1)*2]
        #current_user_pos是一个90x2的二维数组
        self.current_user_pos = current_user_pos
        self.association = association  #当前用户分配给7个基站的情况
        self.K = num_shared             #K表示被MBS和PBS共享的子信道数
        distance = []                   #np.tile(self.macrocell,(90,1))表示对数据self.macrocell,按行复制90次，按列复制1次
        #这样可以方便计算90个用户的位置，分别和macro基站的距离
        #np.tile(self.macrocell,(90,1)),把macrocell坐标复制90次
        mu_relative = current_user_pos - np.tile(self.macrocell,(90,1))  #macrocell user relative position
        mu_dis = np.sqrt(np.sum(np.square(mu_relative),axis=1))          #macrocell user distance
       #mu_dis 是90个用户和M基站之间的距离，是一个长度为90的一维数组，里面的每个元素表示的是一个用户和M基站之间的距离
        distance.append(mu_dis)
        for i in range(NUM_PICO):
            pu_relative = current_user_pos - np.tile(self.picocells[i],(90,1)) #picocell user relative position
            pu_dis = np.sqrt(np.sum(np.square(pu_relative),axis=1))            #picocell user distance
            distance.append(pu_dis)
        distance = np.array(distance)
        #最后distance是一个7x90列的二维数组，其中第一行表示的M基站和90个用户之间的距离
        
        path_loss_macro = (128.1+37.6*np.log10((distance[0]+35)/1000)).reshape((1,90)) #是1行，90列的数组
        path_loss_pico = 140.7+36.7*np.log10((distance[1:7]+10)/1000)  #这里的path_loss_pico是6行，90列的数组
        path_loss = np.concatenate([path_loss_macro, path_loss_pico]) #path_loss 是一个7行，90列的数组
        channel_gain_macro = - path_loss_macro - SHADOW_FADING + ANT_GAIN_MACRO
        channel_gain_pico = -path_loss_pico - SHADOW_FADING + ANT_GAIN_PICO
        channel_gain = np.concatenate([channel_gain_macro, channel_gain_pico]) #channel_gain是一个7行，90列的数组
        G = np.power(10, (channel_gain/10.0))  #总共接收到的功率
        #因此，G是一个7行x90列的数组,里面的每个元素表示用户和每一个相对应基站间的信息增益

        #信道增益是指信号衰减的系数
        #以上部分主要是根据用户到达的位置，计算出其信道增益
        ######################################################


        #######################################################
        #数据率作为奖励
        #该部分主要用来计算用户的数据率
        M = NUM_SUB  #总共子信道的个数
        K = num_shared  #可以共享的子信道个数
        assert K <= M
        # using Partially Shared Deployment 采用部分共享部署机制
        Pm = np.power(10, (P_MAX_MACRO/10.0))          # maximum macrocell transmit power in mW 
        Pp = np.power(10, (P_MAX_PICO/10.0))           #微基站接收到的功率
        Pmc = (Pm-Pp)/(M-K)         #M基站中平均每个独占信道接收到的功率
        Ppc = Pp/K                  #P基站中平均每个共享信道上接收到的功率
        N0 = np.power(10, NOISE_DENSITY/10.0) * SUB_BW * 1000       # noise power in mW   #噪声功率
        
        gamma_macro = {}         # dictionary
        gamma_pico = {}          # dictionary
        # compute SINR for user associated to macrocell
        gamma_macro['exclusive'] = []   #连上M基站的有两种情况，共享信道接入和独占信道接入
        gamma_macro['shared'] = []
        for i in range(NUM_UE):
            gamma1 = Pmc*G[0][i]/N0  #表示与基站M之间相连用户的增益
            gamma_macro['exclusive'].append(gamma1)
            interference = 0      #由于共享信道，信道间的频率是一样的，这样即使连接在M基站上也会和其他的几个基站间存在自干扰现象
            for j in range(1, 7):
                interference += Ppc*G[j][i] 
            gamma2 = Ppc*G[0][i]/(N0 + interference)
            gamma_macro['shared'].append(gamma2)
        
        # compute SINR for user associated to picocell
        for num in range(1,7):
            gamma_pico[num] = []
            for i in range(NUM_UE):
                interference = 0
                for j in range(0,7):
                    if j!=num:
                        interference += Ppc*G[j][i]
                gamma = Ppc*G[num][i]/(N0+interference)
                gamma_pico[num].append(gamma)
        
        R = np.zeros((7,90))  #是一个7x90的二维数组
        #这个是M基站和所有连接用户的数据速率
        R[0] = (M-K)*SUB_BW*1000*np.log(1+np.array(gamma_macro['exclusive'])) + K*SUB_BW*1000*np.log(1+np.array(gamma_macro['shared'])) 
        #剩下的这些是6个P基站所连用户的数据速率
        for num in range(1,7):
            R[num] = K*SUB_BW*1000*np.log(1+np.array(gamma_pico[num]))
            
        num_user_bs = np.zeros((7,))  #[0. 0. 0. 0. 0. 0. 0.]用来保存用户所分配的基站
        #只有用户和基站建立连接，就是association中的值不为0时，才会有数据率，否则没有建立连接，更别提
        for j in range(0,7):
            num_user_bs[j] = np.sum(association[j])   #表示分配到第j个基站的用户个数
        N = num_user_bs

        #统计连接到基站上的用户个数
        rate = np.zeros((NUM_UE,))  #用来表示每个用户的数据传输速率，在本论文中，传输速率是奖励函数，是个长度为90的一维数组
        #rate是一个长度为90 的一维数组，里面的元素表示的是每个用户和基站之间的数据率
        #association在初始化时就是一个7x90的二维数组，是由单位矩阵经过转置得到的
        for i in range(NUM_UE):
            a_index = association[:,i].tolist().index(1) #找到该用户连接的基站(查找值为1，（这里的1由单位矩阵得到的）如果某个用户分配到该基站，则其值为1)的第一索引位置)
            rate[i] = R[a_index][i]/N[a_index]    #连接到该基站的用户平均功率，每个用户的权重是一样的（这里也可以考虑不用除N[a_index]）
              #N[a_inex]表示连接到该基站上的用户个数
        self.scheduling_ptr += 1
        
        end_of_trace = False
        if self.scheduling_ptr >= TOTAL_SCHEDULING_INT:
            end_of_trace = True
            self.scheduling_ptr = 0
            self.trace_idx += 1   #轨迹的个数（也就是情景的个数）
            if self.trace_idx >= len(self.all_user_pos):
                self.trace_idx = 0    
            self.user_pos = self.all_user_pos[self.trace_idx]  #重新从第一个进行选择
        #reward = np.sum(np.log(rate))             # using proportional fairness objective as reward function 
        #返回的状态有信道收益，分配到的相应基站的用户个数，数据速率，是否结束的而标致
        return channel_gain, num_user_bs, rate/MEGA, end_of_trace  #rate in Mbps
