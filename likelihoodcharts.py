import sys

from likelihood2 import *

g_interactive = False
trunk = 'li'

def make_rhc_restarts():
    runs = 100
    suite = TestSuite(0)
    fn = CSequence()
    runner = RHCRunner(fn, {'restarts':5,  'argmax_mode':True, 'max_iters':50, 'max_attempts':50})
        
    curves = make_curve(suite, runner, dict(restarts=[5, 10, 30]), runs=runs, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        tm = curves[params]['time']
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='restarts={} (time={:.1f} sec)'.format(params[0], tm))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
        #p = plt.plot(x, p50, label='restarts={} (time={:.1f} sec)'.format(params[0], tm))
        #plt.fill_between(x, p33 , p66, alpha=0.2, color=p[-1].get_color())        
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-restarts.pdf'.format(trunk))
    if g_interactive:
        plt.show()

def make_sa_params():
    runs = 50
    suite = TestSuite(0)
    fn = CSequence()
    runner = SARunner(fn, dict(schedule = mr.GeomDecay(),  
                            max_attempts = 8, max_iters = 180))
    
    curves = make_curve(suite, runner, dict(schedule__init_temp=[1, 1, 10, 10, 10, 100, 100], schedule__decay=[0.1, 0.5, 0.1, 0.5, 0.7, 0.1, 0.5]), 
                        runs=runs, 
                        is_product=False, cutoff=0.3)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='$T_0$={} $r$={}'.format(*params))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-sa-params.pdf'.format(trunk))
    if g_interactive:
        plt.show()



def make_ga_mate():
    runs = 20
    suite = TestSuite(0)
    fn = CSequence()
    runner = GARunner(fn, dict(max_attempts=8, pop_size=200,  mutation_prob=0.1, elite_dreg_ratio=0.99))
    
    curves = make_curve(suite, runner, dict(pop_breed_percent=[0.5, 0.6, 0.7, 0.8, 0.9]), runs=runs, is_product=False, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='mate %={}'.format(*params))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-sa-breed.pdf'.format(trunk))
    if g_interactive:
        plt.show()


def make_ga_mutation():
    runs = 20
    suite = TestSuite(0)
    fn = CSequence()
    runner = GARunner(fn, dict(max_attempts=8, pop_size=200, pop_breed_percent=0.6,  elite_dreg_ratio=0.99))
    
    curves = make_curve(suite, runner, dict(mutation_prob=[0.02, 0.05, 0.1, 0.3, 0.5, 0.7]), runs=runs, is_product=False, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='mutation prob={}'.format(*params))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-ga-mutation.pdf'.format(trunk))
    if g_interactive:
        plt.show()



def make_mimic_keep():
    runs = 15
    suite = TestSuite(0)
    fn = CSequence()
    runner = MIMICRunner(fn, dict(keep_pct=0.3, pop_size=500, max_attempts=5))
    
    curves = make_curve(suite, runner, dict(keep_pct=[0.1, 0.2, 0.3, 0.4, 0.6]), runs=runs, is_product=False, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='keep%={}'.format(*params))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-mimic-keep.pdf'.format(trunk))
    if g_interactive:
        plt.show()


def make_mimic_pop():
    runs = 15
    suite = TestSuite(0)
    fn = CSequence()
    runner = MIMICRunner(fn, dict(keep_pct=0.3, pop_size=1000, max_attempts=5))
    
    curves = make_curve(suite, runner, dict(pop_size=[200, 400, 600, 1000, 1500, 2000]), runs=runs, is_product=False, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        mean = np.array(curves[params]['mean'])
        std = np.array(curves[params]['std'])
        tm = curves[params]['time']
        
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, mean, label='pop={} (time={:.1f} sec)'.format(params[0], tm))
        #plt.fill_between(x, mean-std , mean+std, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('{}-mimic-pop.pdf'.format(trunk))
    if g_interactive:
        plt.show()

arg = sys.argv[1]
if arg=='restart':
    make_rhc_restarts()
elif arg=='sa':
    make_sa_params()
elif arg=='mate':
    make_ga_mate()
elif arg=='mutation':
    make_ga_mutation()
elif arg=='keep':
    make_mimic_keep()
elif arg=='pop':
    make_mimic_pop()