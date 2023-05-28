

class Candidate(object):
    def __init__(self, action, expression, error):
        self.action = action
        self.expression = expression
        self.error = error
    
        

class SaveBuffer(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.candidates = []

    def num_candidates(self):
        return len(self.candidates)

    def add_new(self, candidate):
        flag = 1
        action_idx = None
        for idx, old_candidate in enumerate(self.candidates):
            if (candidate.action == old_candidate.action).all() and candidate.error < old_candidate.error:  # 如果判断出来和之前的action一样的话，就不去做
                flag = 1
                action_idx = idx
                break
            elif (candidate.action == old_candidate.action).all():
                flag = 0

        if flag == 1:
            if action_idx is not None:
                self.candidates.pop(action_idx)
            self.candidates.append(candidate)
            # 去掉loss最大的
            self.candidates = sorted(self.candidates, key=lambda x: x.error)  # from small to large

        if len(self.candidates) > self.max_size:
            self.candidates.pop(-1)  # remove the last one