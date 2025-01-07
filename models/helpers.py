#!/usr/bin/env python3

import sys

class EarlyStopper:
    def __init__(self, eval_fn, save_fn, load_fn, frequency=1, allow_worse=5):
        self.eval_fn = eval_fn
        self.save_fn = save_fn
        self.load_fn = load_fn
        self.frequency = frequency
        self.allow_worse = allow_worse
        self.worse_left = allow_worse
        self.best_score = -(sys.maxsize - 1)
        self.scores = []

    def __call__(self, epoch):
        done = False
        if epoch % self.frequency == 0:
            result = list(self.eval_fn().values())[0]
            if result <= self.best_score:
                if self.worse_left > 0:
                    self.worse_left -= 1
                else:
                    self.load_fn()
                    done = True
            else:
                self.save_fn()
                self.best_score = result
                self.worse_left = self.allow_worse
        return done
