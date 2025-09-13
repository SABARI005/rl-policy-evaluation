# POLICY EVALUATION

## AIM
To evaluate and compare different policies in the Frozen Lake environment and find the best policy for reaching the goal successfully.

## PROBLEM STATEMENT
In the Frozen Lake environment, an agent must navigate from the start to the goal while avoiding holes. Movements are uncertain due to slipperiness. A policy guides the agent’s actions, but not all policies are effective. The task is to:

Evaluate a given policy (V1) using policy evaluation. Create and test a new policy (V2) to improve performance. Compare both policies based on success rate and rewards. Find the best policy for safely reaching the goal. This helps in identifying the most efficient way to complete the task.

## POLICY EVALUATION FUNCTION
```python
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
      V=np.zeros(len(P))
      for s in range(len(P)):
        for prob,next_state,reward,done in P[s][pi(s)]:
           V[s]+=prob*(reward+gamma *prev_V[next_state]*(not done))
      if np.max(np.abs(prev_V-V))<theta:
        break
      prev_V=V.copy()
    return V
```

## OUTPUT:
#### Policy 1:
![image](https://github.com/user-attachments/assets/1064cb60-e5a3-44c4-8367-5fe99d7cc96f)
![image](https://github.com/user-attachments/assets/fa73d672-a911-44f7-933e-6604402bac72)
![image](https://github.com/user-attachments/assets/97a0b2ef-b42b-4c18-ba84-ac6208d789a6)

#### Policy 2:
<img width="508" height="160" alt="image" src="https://github.com/user-attachments/assets/3219707e-9e5c-46ad-b229-466e21676a74" />

![image](https://github.com/user-attachments/assets/12b129dc-47c4-4f42-a206-7ce1dd4268c6)
![image](https://github.com/user-attachments/assets/cf96c83b-7170-473e-8c05-bf992b98a873)

#### V1>=v2
![image](https://github.com/user-attachments/assets/9d9177dc-0af2-4067-b59a-a53b7dac724a)

## RESULT:
Thus, The Python program to evaluate the given policy is successfully executed.
