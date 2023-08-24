import ivy

def MSE(logits, labels):
    return ivy.mean((labels - logits)**2)

def MSE_back(logits, labels):
    return ivy.jac(MSE)(logits, labels)

def CCE(logits, labels):
    return - ivy.sum(logits * ivy.log(labels))

def CCE_back(logits, labels):
    return ivy.jac(CCE)(logits, labels)

def softmax_CCE(logits, labels):
    e_logits = ivy.exp(logits)
    return CCE(e_logits/ivy.sum(e_logits), labels)

def softmax_CCE_back(logits, labels):
    return ivy.jac(softmax_CCE)(logits, labels)