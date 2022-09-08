import numpy as np
import matplotlib.pyplot as plt
from models.tokenization_bert import BertTokenizer
import torch
from utils import pre_question

with open("question.txt", 'r') as f:
    q = f.read()
    print(q)
# question = "What color is the bench on the ground?"
question = pre_question(q, 30)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
question_input = tokenizer(question, padding='longest', return_tensors="np")
print(question_input)
tokens = tokenizer.decode(question_input['input_ids'][0])
with open('attention.npy', 'rb') as f:
    a = np.load(f)

print(a)
print(tokens)
fig1, axs = plt.subplots(3, 4, figsize=(16, 9), constrained_layout=True)
i = 0
for row in axs:
    for col in row:
        col.bar(tokens.split(), a[i])
        i += 1
# for ax, head in zip(axs, a):
    # ax.plot(head)

fig1.savefig("attentions.png")
plt.clf()
S = torch.from_numpy(a)
KL = torch.mean(S[:, None, :] * (S[:, None, :] / S).log(), dim=2)
mean_kl = KL.mean().item()
print(KL.size())
plt.title(f"Multiheaded Confusion: {mean_kl:.3f}")
plt.imshow(KL, cmap='hot', interpolation='nearest')
plt.savefig("confusion.png")
