"""Pool-stratified distributions, quality–yield trade-offs, and cell-level sorting."""


from scipy.stats import gaussian_kde


from matplotlib.colors import Normalize,LinearSegmentedColormap


from matplotlib.transforms import Bbox


from figure_style import *


def place_labels(ax,points):
    """Place short method IDs without overlap, with explicit leader lines."""
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer(); used=[]
    obstacles=[]
    for collection in ax.collections:
        for xy in collection.get_offsets():
            xp,yp=ax.transData.transform(xy)
            obstacles.append(Bbox.from_bounds(xp-3,yp-3,6,6))
    candidates=[(6,7),(6,-13),(-20,7),(-20,-13),(6,17),(-20,17),(13,-4),(-27,-4),
        (6,-23),(-20,-23),(17,12),(-32,12),(17,-20),(-32,-20),(6,26),(-20,26)]
    for method,x,y in points:
        best=None
        for ox,oy in candidates:
            ann=ax.annotate(IDS[method],(x,y),xytext=(ox,oy),textcoords='offset points',fontsize=8.5,
                arrowprops=dict(arrowstyle='-',lw=.45,color='#5E5E5E'),annotation_clip=False,
                bbox=dict(fc='white',ec='none',alpha=.90,pad=.25))
            ann.update_positions(renderer)
            # Use the text bounding box, not the annotation's leader-line extent.
            tb=Text.get_window_extent(ann,renderer=renderer)
            interior=ax.get_window_extent(renderer)
            score=sum(tb.overlaps(z) for z in used)*100
            score+=sum(tb.overlaps(z) for z in obstacles)*30
            score+=max(0,interior.x0-tb.x0)+max(0,tb.x1-interior.x1)
            score+=max(0,interior.y0-tb.y0)+max(0,tb.y1-interior.y1)
            if best is None or score<best[0]:best=(score,(ox,oy),tb)
            ann.remove()
            if score==0:break
        ox,oy=best[1]
        ann=ax.annotate(IDS[method],(x,y),xytext=(ox,oy),textcoords='offset points',fontsize=8.5,
            arrowprops=dict(arrowstyle='-',lw=.45,color='#5E5E5E'),annotation_clip=False,
            bbox=dict(fc='white',ec='none',alpha=.90,pad=.25))
        ann.update_positions(renderer)
        used.append(Text.get_window_extent(ann,renderer=renderer).expanded(1.60,1.25))

