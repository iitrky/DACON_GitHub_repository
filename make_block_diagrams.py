"""Draws the two block diagrams of the paper (Fig1 system model, Fig2 DACON module)."""
import os
HERE=os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams['font.family']='DejaVu Sans'; plt.rcParams['hatch.color']='#c9c9c9'; plt.rcParams['hatch.linewidth']=0.6
def box(ax,x,y,w,h,t,fc,fs=9,hatch=None):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.03,rounding_size=0.1",fc=fc,ec='#222',lw=1.1,hatch=hatch))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=fs,bbox=dict(fc=fc,ec='none',pad=1.5) if hatch else None)
def arr(ax,x1,y1,x2,y2,c='#222',ls='-'): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.3,color=c,ls=ls))
# Fig 1 system model
fig,ax=plt.subplots(figsize=(11,4.7)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(-0.35,4.4)
box(ax,0.1,1.7,1.7,1.0,'Labelled data\nrepository\n(noisy labels)','#e8eef7')
box(ax,2.2,1.7,1.7,1.0,'Network training\nwith warm-up','#e8eef7')
box(ax,4.3,2.9,2.0,0.95,'Temporal evidence\nmean margin trajectory','#fde9d9',hatch='//')
box(ax,4.3,0.55,2.0,0.95,'Spatial evidence\nreliability-weighted\nneighbour agreement','#e2f0d9',hatch='..')
box(ax,6.8,1.6,1.9,1.2,'DACON\ndependence aware\nclass-wise fusion','#fff2cc')
box(ax,9.1,2.75,1.8,0.95,'Label audit report\nposterior per sample','#ead1dc')
box(ax,9.1,0.6,1.8,0.95,'Repaired training\nlabels','#ead1dc')
arr(ax,1.8,2.2,2.2,2.2); arr(ax,3.9,2.45,4.3,3.3); arr(ax,3.9,1.95,4.3,1.05)
arr(ax,6.3,3.3,6.8,2.55); arr(ax,6.3,1.05,6.8,1.85); arr(ax,8.7,2.45,9.1,3.15); arr(ax,8.7,1.95,9.1,1.1)
ax.annotate('',xy=(3.05,1.65),xytext=(10.0,0.55),arrowprops=dict(arrowstyle='->',lw=1.2,color='#555',ls='--',connectionstyle='arc3,rad=-0.3'))
ax.text(6.4,-0.3,'repaired labels feed the next training epoch',ha='center',fontsize=8.5,color='#444')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'Fig1.png'),dpi=300); plt.savefig(os.path.join(HERE,'Fig1.eps')); plt.close()
# Fig 2 DACON module
fig,ax=plt.subplots(figsize=(11,5.3)); ax.axis('off'); ax.set_xlim(0,11); ax.set_ylim(-0.5,4.8)
ax.add_patch(FancyBboxPatch((2.0,0.3),5.0,4.15,boxstyle="round,pad=0.02,rounding_size=0.15",fc='none',ec='#777',ls='--'))
ax.text(4.5,4.25,'repeated for each class c',ha='center',fontsize=8.5,color='#444')
box(ax,0.1,3.0,1.6,0.9,'mean margin\nper sample','#fde9d9',hatch='//')
box(ax,0.1,0.8,1.6,0.9,'weighted\nagreement','#e2f0d9',hatch='..')
box(ax,2.3,3.0,1.9,0.9,'1D mixture\nposterior of view 1','#fff2cc')
box(ax,2.3,0.8,1.9,0.9,'1D mixture\nposterior of view 2','#fff2cc')
box(ax,2.3,1.95,1.9,0.75,'2D mixture on\nboth views','#f3f3f3')
box(ax,4.7,1.95,2.1,0.75,'within component\ncorrelation and shrink factor','#f3f3f3',8.3)
box(ax,7.3,1.75,1.7,1.2,'dependence aware\nlog odds fusion','#fce4d6')
box(ax,9.3,0.8,1.6,0.9,'clean posterior\nper sample','#ead1dc')
box(ax,9.3,3.0,1.6,0.9,'consensus soft\ntarget','#ead1dc')
arr(ax,1.7,3.45,2.3,3.45); arr(ax,1.7,1.25,2.3,1.25); arr(ax,1.7,3.2,2.3,2.5); arr(ax,1.7,1.5,2.3,2.15)
arr(ax,4.2,2.32,4.7,2.32); arr(ax,6.8,2.32,7.3,2.32); arr(ax,4.2,3.45,7.3,2.8); arr(ax,4.2,1.25,7.3,1.9)
arr(ax,9.0,2.6,9.3,3.3); arr(ax,9.0,2.1,9.3,1.3)
ax.annotate('',xy=(0.9,0.78),xytext=(10.1,0.78),arrowprops=dict(arrowstyle='->',lw=1.1,color='#555',ls='--',connectionstyle='arc3,rad=-0.12'))
ax.text(5.5,-0.45,'posteriors reweight neighbour votes at the next refinement',ha='center',fontsize=8.5,color='#444')
plt.tight_layout(); plt.savefig(os.path.join(HERE,'Fig2.png'),dpi=300); plt.savefig(os.path.join(HERE,'Fig2.eps')); plt.close()
