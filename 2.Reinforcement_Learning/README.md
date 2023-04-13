# 3. Reinforcement Learning

We have learned a method, dynamic programming, to solve Markov decision processes. Markov decision processes provide a formal framework for modeling decision-making problems in which the optimal decision depends on the current state of the environment. Dynamic programming approximates optimal value functions and policies in iterative ways to find optimal decisions.

However, a huge assumption is required to use dynamic programming. That is, the dynamics $p(s',r|s,a)$ should be known. Since the dynamics define an MDP, dynamic programming is called `model-based` programming. In most cases, however, the dynamics of an environment is not known. Therefore, to solve an MDP without the knowledge of the underlying MDP, we instead use a method of *‘learning’*, which is also called `model-free`learning. In this section, we will figure out `reinforcement learning` as a type of model-free learning.

## 3.1 What is Reinforcement Learning?

Reinforcement learning (RL) is a method to solve Markov decision processes (MDPs) without being known the complete information of an environment. Recall that we say MDPs are solved if an optimal value function is found. In dynamic programming method, an optimal value function $V^\ast(s)$ is obtained by using iterative algorithms based on Bellman equations. With an optimal value function, an optimal policy is found by the following equation:

$$
\pi^\ast(s) = \arg\max_a Q^\ast(s,a) \quad \text{and} \quad Q^\ast(s,a) = \sum_{s',r} \, p(s',r|s,a) [r+ \gamma \,V^\ast(s')]
$$

However, being unknown the dynamics $p(s',r|s,a)$ of environment changes the whole algorithm. First, the exact Bellman equations cannot be implemented in such situations with incomplete information. Second, an optimal policy cannot be found from estimating $V^\ast(s)$ because it also requires the dynamics. Therefore, we should estimate optimal value functions or optimal policies in different perspectives.

As we cannot know the complete information of an environment, RL method only requires experiences (samples or observations) to learn optimal policies. In the following methods, Monte Carlo method and Temporal Difference method, an optimal action-value function is estimated from the samples.

Such methods are under a principle of `generalized policy iteration (GPI)`, which alternatively proceeds policy evaluation and policy improvement repeatedly. 

## 3.2 Monte Carlo Method

Monte Carlo methods are ways of solving a given MDP, i.e., finding an optimal action-value functions of an MDP, based on averaging sample returns. To ensure that well-defined returns are available, we define Monte Carlo methods only for *episodic tasks*. In the policy evaluation phase, the method samples and averages returns to estimate the action-value function with the current policy. In the policy improvement phase, the method updates the policy based on the so far estimated action-value function.

In the policy evaluation phase, we try to estimate $Q^\pi(s,a)$ corresponding to a fixed policy $\pi$. It is noticeable that we estimate the action-value function $Q^\pi(s,a)$ instead of the state-value function $V^\pi(s)$. It is because of the way we improve our policy $\pi$ into $\pi'$.

$$
\pi'(s) = \arg\max_a \sum_{s',r}p(s',r|s,a)[r + \gamma V^\pi(s')] = \arg\max_a Q^\pi(s,a)
$$

When we are using RL, we are in a situation without knowing the dynamics of given environment. Therefore, we cannot improve the policy by estimating state-value function $V^\pi(s)$. For this reason, action-value functions are estimated in various manners.

### 3.2.1 Monte Carlo Prediction (Policy Evaluation)

Recall the definition of the action-value function:

$$
q_\pi(s,a) = \mathbb{E}_\pi[G_t\,|\, S_t = s, A_t=a]
$$

Therefore, for each state-action pair $(s,a)$, we can estimate $Q^\pi(s,a)$ by averaging returns obtained from a number of episodes in an environment.

---

<aside>
⚙ **Algorithm (MC Prediction)**

</aside>

> Initialize:
> 
> 
> $Q(s,a)$ arbitrarily for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$, and $Q(terminal, \cdot) = 0$
> 
> $Returns(s,a)$ ← empty list, for all $s \in \mathcal{S}$ and $a \in \mathcal{A}(s)$
> 
> $\pi$ ← arbitrarily $\epsilon$-soft policy (non-empty probabilities)
> 
> Repeat forever (for each episode):
> 
> 1. Generate an episode following (fixed) $\pi$: $S_0, A_0, R_1, S_1, A_1, R_2, \dots, S_{T-1}, A_{T-1}, R_T$
> $G$ ← 0
> 2. Repeat (for each step of episode), $t=T-1, T-2, \dots, 0$:
>      $G$ ← $\gamma G + R_{t+1}$
>      Unless $(S_t, A_t)$ appears in $(S_0,A_0), \cdots, (S_{t-1}, A_{t-1})$:
>           Append $G$ to $Returns(S_t,A_t)$
>           $Q(S_t, A_t)$ ← average($Returns(S_t,A_t)$)
> 3. For each $S_t$ in the episode:
>      $\pi(a|S_t) \leftarrow \begin{cases} 1-\epsilon + \frac{\epsilon}{|\mathcal{A}(S_t)} & \text{if} \, a= \arg\max_a Q(S_t,a) \\ \frac{\epsilon}{|\mathcal{A}(S_t)|} & \text{otherwise} \end{cases}$

---

### 3.2.2 Monte Carlo Control (Policy Improvement)

## 3.3 Temporal Difference Method

Recall the update equation of Monte Carlo methods:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ \, G_t - Q(S_t,A_t)\, \Big]
$$

We should be aware that the entry $(s,a)$ of the $Q$-table is updated only when there is an episode whose trajectory contains the pair. That is, when an agent never experiences a pair $(s,a)$ in the environment, then the $Q$-table would not be updated. Moreover, Monte Carlo methods must wait until the end of the episode to update the table, which might be considered inefficient.

Temporal difference (TD) methods estimate the action-value function by using the estimate of the value function at the next time step to update the estimate of the value function at the current time step. Hence, such a bootstrapping method allows us to update $Q$-table after every time step, which makes TD method more efficient. 

We introduce two types of TD method, Sarsa and Q-learning. They both use bootstrapping, but the underlying theory of estimating $Q$-table is a bit different.

### 3.3.1 Sarsa

We first state the update equation of $Q(s,a)$:

$$
Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ \, R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t) \, \Big]
$$

This update equation is indeed eligible to estimate $Q(s,a)$ because of Bellman expectation equation:

$$
q_\pi(s,a) = \mathbb{E}_\pi\big[R_{t+1}+\gamma q_\pi(S_{t+1},A_{t+1}) \, | \, S_t=s,A_t=a \,\big]
$$

However, because $q_\pi(S_{t+1}, A_{t+1})$ is not known, we use the current estimate $Q(S_{t+1}, A_{t+1})$ instead. When its update is based on an existing estimate, we say that it is a `bootstrapping` method. The bootstrapping method also makes the TD target biased. 

The algorithm of Sarsa is as follows:

---

<aside>
⚙ ************************Algorithm (Sarsa)************************

</aside>

> Algorithm parameters: step size $\alpha \in (0,1]$, small $\varepsilon>0$
> 
> 
> Initialize $Q(s,a)$, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(terminal, \cdot) = 0$
> 
> Repeat for each episode:
> 
> Initialize $S$
> 
> Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
> 
> Repeat for each step of episode:
> 
> Take action $A$, observe $R$, $S'$
> 
> Choose $A'$ from $S'$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
> 
> $Q(S, A) \leftarrow Q(S,A) + \alpha [ \, R + \gamma Q(S', A') - Q(S,A) \, ]$
> 
> $S \leftarrow S'$; $A \leftarrow A'$;
> 
> until $S$ is terminal
> 

---

Notice that the update rule uses every element of 5-tuple $(S_t,A_t,R_{t+1}, S_{t+1},A_{t+1})$, that make up a transition from one state-action pair to the next. This 5-tuple give rise to the name ***Sarsa*** for the algorithm. Notice that the action $A_{t+1}$ in the target $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ is chosen by the same policy $\pi$ which chose the action $A_t$ (and we call it `on-policy method`). 

The quantity $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t,A_t)$ is a sort of error, measuring the difference between the estimated value $Q(S_t,A_t)$ and the better estimate $R_{t+1} + \gamma Q(S_{t+1},A_{t+1})$. This quantity is called the *TD error*, and it arises in various forms throughout reinforcement learning.

> The convergence properties of the Sarsa algorithm depend on the nature of the policy’s dependence on $Q$. For example, one could use $\epsilon$-greedy or $\epsilon$-soft policies. Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state-action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy (which can be arrange, for example, with $\epsilon$-greedy policies by setting $\epsilon=1/t$). [page 129, Reference 2]
> 

### 3.3.2 Q-learning

One of the early breakthroughs in reinforcement learning was the development of an `off-policy` TD control algorithm known as Q-learning (Watkins, 1989), defined by

$$
Q(S_t, A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ \, R_{t+1} + \gamma \max_aQ(S_{t+1}, a) - Q(S_t,A_t) \, \Big].
$$

The update equation can be explained by Bellman optimality equation:

$$
q_\ast(s,a) = \mathbb{E}\big[R_{t+1} + \gamma \max_{a'} q_\ast(S_{t+1},a') \,|\,S_t=s,A_t=a \big]
$$

In this case, the $Q$-table directly approximates the optimal action-value function $q_\ast$ independent of the policy being followed.

---

<aside>
⚙ ************************Algorithm (Q-learning)************************

</aside>

> Algorithm parameters: step size $\alpha \in (0,1]$, small $\varepsilon>0$
> 
> 
> Initialize $Q(s,a)$, for all $s \in \mathcal{S}$, $a \in \mathcal{A}(s)$, arbitrarily except that $Q(terminal, \cdot) = 0$
> 
> Repeat for each episode:
> 
> Initialize $S$
> 
> Repeat for each step of episode:
> 
> Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\varepsilon$-greedy)
> 
> Take action $A$, observe $R$, $S'$
> 
> $Q(S, A) \leftarrow Q(S,A) + \alpha [ \, R + \gamma \max_a Q(S', a) - Q(S,A) \, ]$
> 
> $S \leftarrow S'$
> 
> until $S$ is terminal
> 

---

## 3.4 Exploration vs Exploitation

Exploration and exploitation are two conflicting goals in reinforcement learning. Exploitation means taking the action that is currently believed to be the best based on the current knowledge, whereas exploration means taking a non-greedy action to gain more information about the environment. In other words, exploitation is the act of making the best decision based on current information, and exploration is the act of gathering more information to make better decisions in the future. A good balance between exploration and exploitation is required to find the optimal policy and value function. One of the most popular exploration strategies is the $\epsilon$-greedy strategy, which selects the greedy action with probability $1-\epsilon$ and selects a random action with probability $\epsilon$.
