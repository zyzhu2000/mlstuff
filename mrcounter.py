import mlrose_hiive as mr

class PrintMixin(object):
    def __str__(self):
        sb = []
        for key in self.__dict__:
            sb.append("{key}='{value}'".format(key=key, value=self.__dict__[key]))
 
        return '\n'.join(sb)
    
    def __repr__(self):
        return self.__str__()
    
    
class RHCCounter(PrintMixin):
    def __init__(self):
        self.restarts = 0
        self.attempts = []
        self.total_iterations = 0
        self.histogram = []
        self.curves = []
    
    
    def __call__(self, iteration=None,  attempt=None,  done=None, state=None, fitness=None, curve=None, user_data=None):
        self.total_iterations += 1
        if done:
            self.restarts += 1
            self.attempts.append(attempt-1)
            self.histogram.append(fitness)
            self.curves.append(curve)
        return True
            
        
    

