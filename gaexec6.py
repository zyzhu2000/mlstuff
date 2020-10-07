from cd_ann3 import *

ga = train_ga(pop_size=600, max_iters=2000, early_stopping=False)
save('ga600.pkl', ga)