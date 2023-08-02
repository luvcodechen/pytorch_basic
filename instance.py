def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t
