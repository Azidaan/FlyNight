from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
e=read(OUT/'interaction_effects.json')['effects232'];plt.rcParams.update({'font.family':'Times New Roman','font.size':14})
fig,axes=plt.subplots(1,2,figsize=(12,5.5));labels=['Исходно','Только\nпривязка','Только\nудержание','Оба\nизменения'];keys=['C34|index','R134_temporal|index','C34|history_hold','R134_temporal|history_hold']
for ax,row,title in zip(axes,e,['Исходное правило','Правило согласия']):
 values=[row['conditions'][k] for k in keys];ax.bar(range(4),values,color=['#8457a6' if v<0 else '#31865b' for v in values]);ax.axhline(0,color='black',linewidth=1);ax.set_xticks(range(4),labels);ax.set_title(title);ax.set_ylabel('Балл пары 232');ax.margins(y=.25)
 for i,v in enumerate(values):ax.annotate(f'{v:.3f}'.replace('.',','),(i,v),xytext=(0,6 if v>0 else -17),textcoords='offset points',ha='center',fontsize=13)
 ax.spines[['right','top']].set_visible(False)
fig.suptitle('Отказ 232: совместный эффект привязки и состояния ответа',fontsize=16)
fig.text(.5,.02,'Порог 0 и веса фиксированы. Диагностика на сохранённых картах; общий критерий пяти случаев не пройден.',ha='center',fontsize=12)
fig.tight_layout(rect=[0,.09,1,.92]);fig.savefig(OUT/'interaction_232.png',dpi=160);fig.savefig(OUT/'interaction_232.svg');plt.close(fig)
print('PLOT_SAVED')

# Keep SVG geometry while stripping generator-added trailing whitespace.
svg=OUT/'interaction_232.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
