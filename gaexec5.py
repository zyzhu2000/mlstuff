from cd_ann3 import *

ga = train_ga(pop_size=500, max_iters=2000, early_stopping=False)
save('ga500.pkl', ga)