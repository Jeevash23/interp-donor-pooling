"""Vector schematic of the four pooling locations; no empirical values."""
from common import *
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({'font.size':9,'mathtext.fontset':'dejavusans','pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(6.5,2.75));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
xs=[.32,.51,.70,.89];ys=[.77,.56,.35,.14]
for x,label in zip(xs,['Patch vectors','Final residual','Readout logits','Target probability']):ax.text(x,.96,label,ha='center',va='center',fontsize=8.3)
rows=[['Pool before $F$',r'$\mu$',r'$z_\mu$',r'$A$',r'$p(A)$'],
      ['Pool before $U$',r'$\{a_j\}$',r'$\bar z$',r'$B$',r'$p(B)$'],
      ['Pool before $p$',r'$\{a_j\}$',r'$\{z_j\}$',r'$C$',r'$p(C)$'],
      ['Pool after $p$',r'$\{a_j\}$',r'$\{z_j\}$',r'$\{U(z_j)\}$',r'$\overline{p(U(z_j))}$']]
for r,(y,labels) in enumerate(zip(ys,rows)):
    ax.text(.015,y,labels[0],ha='left',va='center',fontsize=8.7)
    for c,(x,lab) in enumerate(zip(xs,labels[1:])):
        box=FancyBboxPatch((x-.077,y-.066),.154,.132,boxstyle='round,pad=0.009,rounding_size=0.012',linewidth=.7,edgecolor='#0072B2' if c==r else '#AAB3BB',facecolor='#E6F1F8' if c==r else '#FAFAFA');ax.add_patch(box)
        ax.text(x,y,lab,ha='center',va='center',fontsize=9 if not (r==3 and c==3) else 8.1)
        if c<3:ax.annotate('',(xs[c+1]-.091,y),(x+.09,y),arrowprops={'arrowstyle':'->','color':'#7A838B','lw':.8})
    if r<3:
        ax.text(.99,(ys[r]+ys[r+1])/2,['$P_p$','$N_p$','$S_p$'][r],ha='right',va='center',fontsize=9,color='#005580')
fig.tight_layout(pad=.1)
dest=ROOT/'results/figures';dest.mkdir(parents=True,exist_ok=True)
fig.savefig(dest/'pooling-boundaries.pdf',bbox_inches='tight');fig.savefig(dest/'pooling-boundaries.png',dpi=200,bbox_inches='tight')
