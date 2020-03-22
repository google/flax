# Node classification using a Graph Neural Network (GNN)
This example trains a simple graph neural network (GNN) for semi-supervised
node classification on Zachary's karate club (Wayne W. Zachary, "An Information
Flow Model for Conflict and Fission in Small Groups," Journal of Anthropological
Research, 1977).

Zachary's karate club is often used as a "hello world" example for network
analysis. The graph describes the social network of members of a university
karate club, where an undirected edge is present if two members interact
frequently outside of club activities. The club famously split into two parts
due to a conflict between the president of the club (John A.) and the part-time
karate instructor (Mr. Hi).

This example is adapted from https://arxiv.org/abs/1609.02907 (Appendix A). We
classify nodes based on the student-teacher assignment (John A. or Mr. Hi) in
a semi-supervised setting. During training, only the labels for John A.'s and
Mr. Hi's node are provided, while all other club members are unlabeled.

### Example output

```
iteration: 1, loss: 1.2327, accuracy: 88.24
iteration: 2, loss: 0.4691, accuracy: 97.06
iteration: 3, loss: 0.2067, accuracy: 100.00
```

### How to run

`python train.py`