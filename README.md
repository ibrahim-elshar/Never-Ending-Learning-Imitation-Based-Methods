# Navigate, Fetch and Find - Imitation Based Methods

### CMU 10-703 Project

We work in the [Never-Ending Learning Gym Environment] (https://github.com/eaplatanios/nel_framework),  which is an endless grid-world with an agent collecting valuable items and tools. 
Collecting rewards efficiently involves taking the shortest path to collect items  and having a high level plan that guides the agent and prioritize where to go and what items to collect. 

It is important because we believe that it can be related to real life problems which involve rescuing humans from various hazards. For example, in fire
situations, robots can be trained to rescue people trapped in buildings/cities burning in fire. Other examples include searching for survivors of earth quakes, Tsunamis, or sinking ships.

We propose a greedy heuristic and implement Deep Q-networks (DQN), 
imitation learning and Deep Q-learning from Demonstrations (DQfD).

Our proposed *Greedy Heuristic policy* is near optimal in most
situations. Our experiments show that Greedy Heuristic outperforms all other methods.