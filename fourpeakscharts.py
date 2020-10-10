import matplotlib
matplotlib.use('pdf')
from fourpeaks2 import *

g_interactive = False

def make_4peaks_restarts():
    runs = 10
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = RHCRunner(fn, {'restarts':5,  'argmax_mode':False})
        
    curves = make_curve(suite, runner, dict(restarts=[5, 10, 20]), runs=runs, extend=True)
    plt.figure()
    
    for params in curves:
        p50 = curves[params]['p50']
        p33 = curves[params]['p33']
        p66 = curves[params]['p66']
        x = np.arange(1, len(p50)+1)
        p = plt.plot(x, p50, label='restarts={}'.format(params[0]), alpha=0.8)
        #plt.fill_between(x, p33 , p66, alpha=0.2, color=p[-1].get_color())
    #plt.title('Effect of Restarts')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Fitness Score', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig('fpeaks-restarts.pdf')
    plt.show()

def make_sa_params():
    runs = 100
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = SARunner(fn, dict(schedule = mr.GeomDecay(),  
                            max_attempts = 8, max_iters = 1000))
    
    curves = make_curve(suite, runner, dict(schedule__init_temp=[2, 2, 2, 0.5, 5], schedule__decay=[0.7, 0.8, 0.9,  0.5, 0.999]), 
                        runs=runs, is_product=False, extend=True)
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
    plt.savefig('fpeaks-sa-params.pdf')
    plt.show()


def make_ga_mate():
    runs = 10
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = GARunner(fn, dict(max_attempts=50, pop_size=200,   elite_dreg_ratio=0.9))
    
    curves = make_curve(suite, runner, dict(pop_breed_percent=[0.5, 0.6, 0.75, 0.9]), runs=runs, is_product=False)
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
    plt.savefig('fpeaks-sa-breed.pdf')
    plt.show()

def make_ga_mutation():
    runs = 10
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = GARunner(fn, dict(max_attempts=50, pop_size=200,   elite_dreg_ratio=0.9))
    
    curves = make_curve(suite, runner, dict(mutation_prob=[0.01, 0.1, 0.5, 0.6, 0.7]), runs=runs, is_product=False)
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
    plt.savefig('fpeaks-sa-mutation.pdf')
    plt.show()


def make_mimic_keep():
    runs = 10
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = MIMICRunner(fn, dict(keep_pct=0.2, pop_size=2000, max_attempts=50))
    #, 0.1, 0.2, 0.3, 0.6
    curves = make_curve(suite, runner, dict(keep_pct=[0.05]), runs=runs, is_product=False)
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
    plt.savefig('fpeaks-mimic-keep.pdf')
    if g_interactive:
        plt.show()


def make_mimic_pop():
    runs = 10
    suite = TestSuite(0)
    fn = FourPeaks()
    runner = MIMICRunner(fn, dict(keep_pct=0.2, pop_size=2000, max_attempts=50))
    
    curves = make_curve(suite, runner, dict(pop_size=[200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]), runs=runs, is_product=False)
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
    plt.savefig('fpeaks-mimic-pop.pdf')
    if g_interactive:
        plt.show()

#make_4peaks_restarts()
#make_sa_params()
#make_ga_mate()
#make_ga_mutation()
make_mimic_keep()
#make_mimic_pop()