from cd_ann3 import *

ga = train_ga(pop_size=200, max_iters=2000, early_stopping=False)
save('ga200.pkl', ga)