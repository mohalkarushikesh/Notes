what is RL?
-> Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions and receiving rewards or penalties.
-> The goal is to maximize the total reward over time by learning the best actions to take in different situations.
-> It's like training a dog with treats for good behavior!

Blog : Simple RL with Tensorflow by arthur juliani, 2016 (https://awjuliani.medium.com/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0)
R. paper: RL introduction by richard sutton (1998) (https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

Bellman Equation is based on the principle of optimality, which states that the optimal value of a state can be computed based on the immediate reward and the optimal value of the next state.
    concepts: 
        s-State
        a-Action
        R-Reward
        gamma-Discount
        Bellman Equation img: ![{5DD3D2F9-F15D-4D46-AAD1-7058D777DDAB}](https://github.com/user-attachments/assets/342cb01f-334e-4b61-bab7-acd710bf4702)
        R. paper: The theory of dynamic programming by richard bellaman (1954)
        bellaman equation concept img: ![image](https://github.com/user-attachments/assets/18c92c06-b52a-460c-83be-3c3184ca35ee)
The plan 

Markon Decision process(MDP) 
    Deterministic Search
    Non-Deterministic Search
    A Markov process is a random process indexed by time, and with the property that the future is independent of the past, given the present. Markov processes, named for Andrei Markov, are among the most important of all random processes
    A Markov Decision Process (MDP) is a mathematical model that helps make decisions in dynamic systems. It's used when decisions are random or controlled by a decision maker. 
    updated bellaman equation img: ![image](https://github.com/user-attachments/assets/a8979e32-ba6e-4bd0-af25-a1566d74a735)
    R. paper: A survey of Applications of Markov Decision process by D.J.Wite (1993)

Policy vs Plan
Adding living penalty
    
Q-learning Intution
    
Temporal diff
Q-learning Visualization
