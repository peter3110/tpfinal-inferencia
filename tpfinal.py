import numpy as np
import matplotlib.pyplot as plt
import math, copy
from scipy.stats.kde import gaussian_kde
# Preguntas:
# 1) -alpha no diferencia cuando y[i] = 1 o y[i] >> 1 al medir prob. de salto
# 2) -en los extremos, como sampleo theta_j_tilde ?
#    -los thetas de los extremos tienen densidad infinita en comparacion con el resto
#    -samplear theta_j_tilde entre theta[j-1] y theta[j+1] o en todo el intervalo?
#    siempre de forma uniforme o con una normal?
# 3) -alpha y likelihood estan bien? una normal , centrada en cant_thetas_in_interval y con
#    un parametro de variacion sigma ? esta bien el parametro sigma = 1 o .X ?
#    esta bien que el alpha no diferencie si en un mismo intervalo, dos thetas[j] 
#    consecutivos estan muy cerca o separados uno del otro? (esto creo que no deberia 
#    afectar...)

# Code
def count_thetas_in_interval(t, theta):
    ans = [0] * (len(t) - 1)
    for i in range(0,len(t)-1):
        for j in range(0, len(theta)):
            if t[i] <= theta[j] and theta[j] < t[i+1]:
                ans[i] = ans[i] + 1
    return ans
def pdf_normal_yi(t,y,c,i,theta):
    ans = 1.0 / (sigma * math.sqrt( 2 * math.pi ) )
    ans = ans * math.exp( -.5 * ( (y[i] - c[i] ) / sigma) ** 2 )
    return ans
def pdf_normal_y(t,y,theta):
    c = count_thetas_in_interval(t,theta)
    ans = 0.0
    for i in range(0,len(y)):
        ans = ans + math.log(pdf_normal_yi(t,y,c,i,theta), math.e)
    return math.exp(ans)
def calculate_theta_j(t, theta, j):
    #print theta, ' ' , j
    if j > 0 and j < len(theta) - 1:
        return (theta[j] - theta[j-1]) * (theta[j+1] - theta[j])
    elif j==0:
        return theta[j+1] - theta[j]
    elif j==len(theta)-1:
        return theta[j] - theta[j-1]
def alpha(t,y,theta,j,theta_j_tilde):
    theta_next    = theta[:]
    theta_next[j] = theta_j_tilde
    log_likelihood = math.log( pdf_normal_y(t,y,theta_next), math.e) - \
                     math.log( pdf_normal_y(t,y,theta), math.e)
    log_prior = math.log( calculate_theta_j(t,theta_next,j) , math.e) - \
                math.log( calculate_theta_j(t,theta, j), math.e) 
    log_ans = log_likelihood + 1 * log_prior  # NOTA: 1 = factor importancia del prior (?)
    ans = math.exp(log_ans)
    return min(ans, 1)
def sample_theta_j(t,theta,j):
    if j > 0 and j < len(theta)-1:
        return np.random.uniform(theta[j-1],theta[j+1])
    # Parametro arbitrario (?)
    elif j==0:
        return np.random.uniform(min(t),theta[j]) 
    return np.random.uniform(theta[j],max(t))
def move_theta(niter_theta, t, y, theta, j):
    for i in range(niter_theta):
        theta_j_tilde = sample_theta_j(t,theta,j)
        p = alpha(t,y,theta,j,theta_j_tilde)
        U = np.random.uniform(0,1)
        if U < p:
            theta[j] = theta_j_tilde
    return
def test(t,y,k, niter = 1000, niter_theta=100,with_dimensional_jump = False):
    ax = np.linspace( min(t), max(t), 100 )
    ay = MH(t,y,k,niter,niter_theta,with_dimensional_jump = with_dimensional_jump)
    theta_j_buscado = [i for i in range(1,k-1)]
    for theta_j in theta_j_buscado:
        datax = [ay[theta_j][i] for i in range(0,len(ay[theta_j])-1)]
        kde = gaussian_kde( datax )
        title = "k = " + str(k) + ", t = {" + str(t[0]); 
        for i in range(1,len(t)): 
            title = title + ', ' + str(t[i]);
        title = title + "}"
        plt.plot( ax, kde(ax) )
        plt.title(title, position = (.5,1.05))    
def MH(t,y,k, niter,niter_theta=100,nchains=1,nburn=0,with_dimensional_jump = False):
    chain = [[]] * k
    theta = sorted([np.random.uniform(min(t), max(t)) for i in range(k)])
    for i in range(niter):
        jump_prob = np.random.uniform(0,1)
        jump_birth_prob = birth_prob(k)
        jump_death_prob = death_prob(k)
        jump_theta_prob = 1.0 - jump_death_prob - jump_birth_prob
        if not with_dimensional_jump or jump_prob <= jump_theta_prob:
            j = np.random.randint(0,k)
            move_theta(niter_theta,t,y,theta,j)
            chain[j] = chain[j] + [theta[j]]
        elif jump_prob > jump_theta_prob and jump_prob <= 1.0 - jump_death_prob:
            # birth
            a = 1
        else:
            # death
            a = 1
    return chain
    
    
# Agregando saltos de dimensionalidad...
def prior_k(k_prior_lambda,k):
    return math.exp(-k_prior_lambda) * (k_prior_lambda ** (k-1)) / math.factorial(k-1)
def birth_prob(k): 
    return 0.0
def death_prob(k): 
    return 0.0
kMax = 20; k_prior_lambda = 5;
x = [i for i in range(1,kMax)]; y = [prior_k(k_prior_lambda,z) for z in x];
plt.bar(x,y)

# Tests:
sigma = .6
y1 = [1,1,1,1,1,1,1,1,1,1,1,1]
t1 = [2,4,6,8,10,12,14,16,18,20,22,24,25]
t2 = [6,6.5,6.9,7,7.5,8,9,9.5,10,20,21,22,25]
t3 = [6,6.5,6.9,7,10,20,20.5,21,22,22.5,23,24,25]
t4 = [6,6.5,6.9,7,10,15,20.5,21,22,22.5,23,24,25]
t5 = [6,6.5,6.9,7,10,15,15.5,21,22,22.5,23,24,25]

test(t=t2,y=y1,k=8,with_dimensional_jump=True)
test(t=t3,y=y1,k=12, with_dimensional_jump=False)




#######################################################
# Testear funcion alpha (para theta_j)
# count_thetas_in_interval(t,th)
# th = [4.5,7,8,20,21,25,36]
# j  = 2; tl = th[j-1]; tr = th[j+1];
# x  = np.linspace(tl+.01, tr-.01, (tr - tl)*10)
# yx  = [alpha(t,y,th,j,x[i]) for i in range(0,len(x))]
# plt.plot(x,yx)

# Testear funcion pdf_normal_y : P(y_1..y_n | theta)
# count_thetas_in_interval(t,ths[100])
# ths = [[4.5,7,8,z,21,25,36] for z in np.linspace(8,21,(21-8)*10)]
# x  = np.linspace(tl+.01, tr-.01, (tr - tl)*10)
# yx = [pdf_normal_y(t,y,ths[i]) for i in range(0,len(ths))]
# plt.plot(x,yx)
#######################################################
