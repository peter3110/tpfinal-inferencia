import numpy as np
import matplotlib.pyplot as plt
import math, copy
from scipy.stats.kde import gaussian_kde

# Constantes
kMax = 30; k_prior_lambda = 10; c = 0.45;
factor_prior = .9; delta_intervalo_medicion = 1.7;
error = .001
casi_cero = .001

# Funciones
def count_thetas_in_interval(theta,i):
    ans = 0
    for j in range(0, len(theta)):
        if theta[i] - delta_intervalo_medicion < theta[j] and  \
            theta[j] < theta[i] + delta_intervalo_medicion:
            ans = ans + 1
    return ans
def index_nearest_theta_func(theta,x):
    ans = 0
    mn = (theta[0] - x) ** 2
    for i in range(len(theta)):
        tmp = (theta[i] - x) ** 2
        if tmp < mn:
            mn = tmp
            ans = i
    return ans
def calculate_theta_j(t, theta, j):
    if j > 0 and j < len(theta) - 1:
        ans = (theta[j] - theta[j-1]) * (theta[j+1] - theta[j])
    elif j==0:
        ans = theta[j+1] - theta[j]
    elif j==len(theta)-1:
        ans = theta[j] - theta[j-1]
    return max(casi_cero,ans)
def pdf_normal_ti(t,theta,y,i,x):
    if i < len(theta): index_nearest_theta = i
    else: index_nearest_theta = index_nearest_theta_func(theta,t[i])
    c_i = count_thetas_in_interval(theta,index_nearest_theta)
    if y[i] - c_i != 0: 
        sigma_i = error + 1.0 / ((y[i] - c_i) ** 2)
    else : 
        sigma_i = error
    ans = 1.0 / (sigma_i * math.sqrt( 2 * math.pi ) )
    ans = ans * math.exp( -.5 * ( (x - theta[index_nearest_theta] ) / sigma_i ) ** 2 )
    return max(casi_cero,ans)
def pdf_normal_t(t,y,theta):
    #print theta
    ans = 0.0
    for i in range(0,len(y)):
        ans = ans + math.log(pdf_normal_ti(t,theta,y,i,t[i]), math.e)
    return math.exp(ans)
def alpha_theta(t,y,theta,j,theta_j_tilde):
    theta_next = theta[:]
    theta_next[j] = theta_j_tilde # theta puede quedar desordenado
    log_likelihood = math.log( pdf_normal_t(t,y,theta_next), math.e) - \
                     math.log( pdf_normal_t(t,y,theta), math.e)
    log_prior = math.log( calculate_theta_j(t,theta_next,j) , math.e) - \
                math.log( calculate_theta_j(t,theta, j), math.e) 
    log_ans = log_likelihood + factor_prior * log_prior  # NOTA: 1 = factor importancia del prior (?)
    ans = math.exp(log_ans)
    return min(ans, 1)
def sample_theta_j(t,theta,j):
    if j > len(t) - 1: return np.random.uniform(t[0],t[len(t)-1])
    if j == len(theta) - 1: return np.random.uniform(theta[j-1],max(t))
    if j == 0: return np.random.uniform(min(t),theta[j+1]) 
    return np.random.uniform(theta[j-1],theta[j+1])    
def move_theta(niter_theta, t, y, theta, j):
    for i in range(niter_theta):
        theta_j_tilde = sample_theta_j(t,theta,j) # puede devolver un num q no quede ordenado en theta
        p = alpha_theta(t,y,theta,j,theta_j_tilde)
        U = np.random.uniform(0,1)
        if U < p:
            # theta queda desordenado para valores >= sum(t)
            theta[j] = theta_j_tilde
    return
def sample_interval_j(t,theta,j):
    if j == 0: return np.random.uniform(min(t)-delta_intervalo_medicion * 2, theta[j])
    if j == len(theta): return np.random.uniform(theta[j-1],max(t)+delta_intervalo_medicion * 2)
    return np.random.uniform(theta[j-1],theta[j])
def prior_k_prob(k_prior_lambda,k,theta):
    p_k = math.exp(-k_prior_lambda) * (k_prior_lambda ** (k-1)) / math.factorial(k-1)
    temp = math.factorial(k)
    for i in range(1,len(theta)):
        temp *= (theta[i] - theta[i-1]) / (theta[len(theta)-1] - theta[0])
    return temp * p_k
def birth_prob(k,theta): 
    if k == kMax: return 0.0
    return c * min(1, prior_k_prob(k_prior_lambda,k+1,theta) / prior_k_prob(k_prior_lambda,k,theta))
def death_prob(k,theta): 
    if k <= 2: return 0.0
    return c * min(1, prior_k_prob(k_prior_lambda,k,theta) / prior_k_prob(k_prior_lambda,k+1,theta))    

def alpha_birth(t,y,theta,theta_next):
    # comparar probabilidad de theta vs theta_next donde len(theta) < len(theta_next)
    
    return 0.5
def alpha_death(t,y,theta_next):
    # comparar probabilidad de theta vs theta_next donde len(theta) > len(theta_next)
    
    return 0.5
    
def MH(t,y,k, niter,niter_theta,with_dimensional_jump,nchains=1,nburn=0):
    chain_theta = [[]] * k
    chain_k = []
    theta = sorted([np.random.uniform(min(t), max(t)) for i in range(k)])
    for i in range(niter):
        # Probs de cada tipo de salto
        jump_prob = np.random.uniform(0,1)
        jump_birth_prob = birth_prob(k,theta)
        jump_death_prob = death_prob(k,theta)
        jump_theta_prob = 1.0 - jump_death_prob - jump_birth_prob
        
        if not with_dimensional_jump or jump_prob <= jump_theta_prob:
            j = np.random.randint(0,k)
            move_theta(niter_theta,t,y,theta,j)
            chain_theta[j] = chain_theta[j] + [theta[j]]
        elif jump_prob > jump_theta_prob and jump_prob <= 1.0 - jump_death_prob:
            # birth
            U = np.random.uniform(0,1)
            j = np.random.randint(0,k+1) # interval of birth, in [0..k]
            theta_j_new = sample_interval_j(t,theta,j)
            theta_next = theta[:]
            theta_next = np.insert(theta_next,j,theta_j_new)
            if U < alpha_birth(t,y,theta,theta_next):
                theta = theta_next[:]
                if len(chain_theta) < len(theta): chain_theta = chain_theta + [[]]
                k = k + 1
            chain_k = chain_k + [k]
        else:
            # death
            j = np.random.randint(0,k) # theta_j dies
            theta_next = theta[:]
            theta_next = np.delete(theta_next,j)
            U = np.random.uniform(0,1)
            if U < alpha_death(t,y,theta_next):
                theta = theta_next[:]
                k = k - 1
            chain_k = chain_k + [k]
    return (chain_theta,chain_k)

def test_theta(t,y,k,theta_j_buscado,ay,with_dimensional_jump,niter = 1000, niter_theta=100):
    ax = np.linspace( min(t) - 4, max(t) + 4, 100 )
    theta_mle = []
    for theta_j in theta_j_buscado:
        datax = [ay[theta_j][i] for i in range(0,len(ay[theta_j])-1)]
        datax = [ay[theta_j][i] for i in range(0,len(ay[theta_j])-1)]
        kde = gaussian_kde( datax )
        title = "k = " + str(k) + ", t = {" + str(t[0]); 
        for i in range(1,len(t)): 
            title = title + ', ' + str(t[i]);
        title = title + "}"
        plt.plot( ax, kde(ax) )
        plt.title(title, position = (.5,1.05)) 
        n, b = np.histogram(datax)
        theta_mle = theta_mle + [ b[np.argmax(n)] ]
    return theta_mle
def get_t(t,t_length,t_min,t_max,y,theta):
    rmse = 0.0
    x = np.linspace( t_min, t_max, 500 )
    for i in range(t_length):
        y = [pdf_normal_ti(t,theta,y,i,x[j]) for j in range(len(x))]
        #plt.plot(x,y)
        i_max = np.argmax(y)
        rmse = rmse + (t[i] - x[i_max])**2
        print t[i],' ',x[i_max]
    print 'RMSE: ', math.sqrt(rmse)
    return
def plot_alpha(t,y,theta,j):
    x = np.linspace(t[0],t[len(t)-1],1000)
    y = [alpha_theta(t,y,theta,j,z) for z in x]
    plt.plot(x,y)


# Tests
y1 = [1,1,1,1,1,1,1,1,1,1,1,1]
t1 = [2,4,6,8,10,12,14,16,18,20,22,24,26]
t2 = [6,6.5,6.9,7,7.5,8,9,9.5,10,20,21,22,25]
t3 = [6,6.5,6.9,7,10,20,20.5,21,22,22.5,23,24,25]
t4 = [2,10,20,40,60,61,62,80,90,91,92,120,200]


# Test t1
t = t1; y = y1; k = 13; niter = 1000; niter_theta = 100; with_dimensional_jump = True; # o False
ay1,ak1 = MH(t,y,k,niter,niter_theta,with_dimensional_jump)
theta1 = test_theta(t,y,k,[i for i in range(0,k)],ay1,with_dimensional_jump)
get_t(t,len(t),min(t),max(t),y,theta1)
plt.hist(ak1)

#Test t2
t = t2; y = y1; k = 13; niter = 1000; niter_theta = 100; with_dimensional_jump = False;
ay2,ak2   = MH(t,y,k,niter,niter_theta,with_dimensional_jump)
ay22,ak22 = MH(t,y,k+2,niter,niter_theta,with_dimensional_jump)
theta2 = test_theta(t,y,k,[i for i in range(k)],ay2,with_dimensional_jump)
theta2 = test_theta(t,y,k+2,[i for i in range(k+2)],ay22,with_dimensional_jump)
get_t(t,len(t),min(t),max(t),y,theta2)

#Test t4: nota: el RMSE se dispara pero debido a theta_12 y theta_13
t = t4; y = y1; k = 13; niter = 1000; niter_theta = 100; with_dimensional_jump = False;
ay4,ak4   = MH(t,y,k,niter,niter_theta,with_dimensional_jump)
ay44,ak44 = MH(t,y,k+2,niter,niter_theta,with_dimensional_jump)
theta4 = test_theta(t,y,k,[i for i in range(k)],ay4,with_dimensional_jump)
theta4 = test_theta(t,y,k+2,[i for i in range(k+2)],ay44,with_dimensional_jump)
get_t(t,len(t),min(t),max(t),y,theta4)


#############################################################
# More tests...
test_theta(t2,y1,18,[i for i in range(18)])
test_theta(t2,y1,20,[i for i in range(12)],niter=3000)
test_theta(t4,y1,18,[1,4,8,12,18])
test_theta(t4,y1,18,[i for i in range(12)])
test_theta(t3,y1,12)
test_theta(t3,y1,20)
test_theta(t4,y1,12,niter=3000)
test_theta(t4,y1,20)

plot_alpha(t1,y1,theta2,7)
plot_alpha(t2,y1,theta2,10)


