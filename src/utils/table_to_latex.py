
tableAND = '''X	Avg	Std

bipolar	8.066666666666666	0.249443825785'''

tableOR = '''X	Avg	Std

bipolar	9.5	0.5'''

t1 = [o.split('\t') for o in tableAND.split('\n')[2::2]]
t2 = [o.split('\t') for o in tableOR.split('\n')[2::2]]

t1 = [[x[0], float(x[1]), float(x[2])] for x in t1]
t2 = [[x[0], float(x[1]), float(x[2])] for x in t2]

#res = '\n'.join(['[ -{0:s}; {0:s} ] & {1:.4f} $\pm$ {2:.4f} & {3:.4f} $\pm$ {4:.4f}\\\\\\'.format(x[0],x[1],x[2],y[1],y[2]) for x,y in zip(t1,t2)])
res = '\n'.join(['{0:s} & {1:.4f} $\pm$ {2:.4f} & {3:.4f} $\pm$ {4:.4f}\\\\\\'.format(x[0],x[1],x[2],y[1],y[2]) for x,y in zip(t1,t2)])

print(res)