Here are the steps to run my code:

1. Dwonload the code from: https://github.com/zyzhu2000/mlstuff
2. Run the following in order to generate the charts I use to tune parameters:
   a. python fourpeakscharts.py
   b. python likelihoodcharts.py
   c. python trapcharts.py
  
3. Run the following in order to generate the data files:
   python  fourpeaks2.py 20 40 50 60 70
   python  likihood2.py 16 32 48 70 80
   python trap2.py  
   
   These will generate a bunch of .pkl pickle files for the next steps
   
4. Run the following to generate the charts and tables:
   a. go to the current directory under a terminal window
   b. run jupyter notebook from terminal window
   c. open summarize.ipynb and select Kernel: Restart and run all

5. Run the following to train the neural network and generate reports
   a. go to the current directory under a terminal window
   b. run jupyter notebook from the terminal window
   c. open nn3.ipynb and select Kernel: Restart and run all
   
References:

1. mlrose--hiive (https://pypi.org/project/mlrose-hiive/)  
   Note: Please do not download. I use a private copy of heavily modified mlrose-hiive that is included in my code repository at https://github.com/zyzhu2000/mlstuff

2. Scikit-learn: https://scikit-learn.org/stable/index.html

   